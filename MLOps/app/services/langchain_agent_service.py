import os
import json
import requests
from urllib.parse import quote
from typing import List, Dict, Any

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma

@tool
def elastic_search(region: str, categories: List[str]) -> list:
    """
    엘라스틱 검색 도구

    사용자가 원하는 지역(region)과 카테고리(categories)를 입력받아,
    해당 조건에 부합하는 장소들의 UUID 리스트를 반환합니다.
    만약 정확한 카테고리가 입력되지 않더라도, 유사한 카테고리를 추출하여 검색에 활용합니다.

    Args:
        region (str): 사용자가 명시한 지역명 (예: '강남', '홍대', '성수동', '이태원로', '잠실역')
        categories (list of str): 최소 1개, 최대 3개의 장소 카테고리 (예: '전시관', '기념관', '전문매장/상가', '5일장', '특산물판매점', '백화점', '상설시장', '문화전수시설', '문화원', '서양식', '건축/조형물', '음식점&카페', '박물관', '컨벤션센터', '역사관광지', '복합 레포츠', '공예/공방', '이색음식점', '영화관', '산업관광지', '중식', '문화시설', '쇼핑', '수상 레포츠', '관광지', '육상 레포츠', '학교', '관광자원', '스키(보드) 렌탈샵', '대형서점', '휴양관광지', '외국문화원', '자연관광지', '레포츠', '한식', '일식', '도서관', '체험관광지', '카페/전통찻집', '면세점', '공연장', '미술관/화랑')

    Returns:
        list: 조건에 부합하는 장소의 UUID 리스트 (예: ['uuid1', 'uuid2', ...])

    Example:
        >>> elastic_search('종로', ['음식점&카페', '문화시설'])
        ['a1b2c3', 'd4e5f6']
    """
    if not region or not categories:
        return []

    print(f"--- 엘라스틱 검색: region={region}, categories={categories} ---")

    region_encoded = quote(region)
    category_encoded = '&'.join([f"categories={quote(cat)}" for cat in categories])

    url = f"http://15.164.50.188:9201/api/place/search/llm-tool?region={region_encoded}&{category_encoded}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response_json = response.json()
        uuid_list = response_json.get('uuids', [])
        print(f"--- 엘라스틱 검색 결과: {len(uuid_list)}개 장소 ---")
        return uuid_list
    except requests.exceptions.RequestException as e:
        print(f"엘라스틱 검색 API 호출 오류: {e}")
        return []


class LangchainAgentService:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다.")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 임베딩 모델 초기화
        self.model_bge = HuggingFaceEmbeddings(
            model_name="upskyy/bge-m3-korean",
            model_kwargs={'device': 'cpu'},  # 'cuda' 사용 가능 시 변경
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # ChromaDB 로드
        chroma_db_path = "/home/ubuntu/MLOps/data/chroma_db_bge"
        if not os.path.exists(chroma_db_path):
            raise FileNotFoundError(f"ChromaDB 경로를 찾을 수 없습니다: {chroma_db_path}")
        self.chroma_bge = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=self.model_bge
        )
        print("--- ChromaDB 로드 완료 ---")
        
        self.tools = [elastic_search, self._create_search_with_filtering_tool()]
        self.prompt = self._create_prompt_template()
        self.user_memories = {}

    def _create_search_with_filtering_tool(self):
        @tool
        def search_with_filtering(query: str, place_uuids: List[str]) -> List[Dict[str, Any]]:
            """
                임베딩 검색 도구

                입력받은 UUID 리스트와 사용자의 쿼리문(query)을 기반으로,
                메타데이터 필터링 및 임베딩 유사도 검색을 수행하여 관련 장소 정보를 반환합니다.

                Args:
                    query (str): 사용자가 입력한 쿼리문 (예: '홍대에서 감성 카페 갔다가 전시 보는 코스 추천')
                    place_uuids (list of str): 검색 대상 장소의 UUID 리스트

                Returns:
                    list: 임베딩 유사도 기반으로 필터링된 장소 정보 리스트

                Example:
                    >>> search_with_filtering('감성적인 카페', ['uuid1', 'uuid2'])
                    [{'name': 'OO레스토랑', 'address': '서울...', ...}, ...]
            """
            if not query or not place_uuids:
                return []
            
            print(f"--- 필터링 및 임베딩 검색: query='{query}', UUIDs={len(place_uuids)}개 ---")

            retriever = self.chroma_bge.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 10,  # 성능을 위해 반환 개수 제한
                    "filter": {"uuid": {"$in": place_uuids}}
                }
            )
            documents = retriever.invoke(query)
            return [doc.metadata for doc in documents]

        return search_with_filtering

    def _create_prompt_template(self):
        template = """
        당신은 서울 코스 추천 챗봇입니다.

        아래의 도구들을 반드시 순서대로 사용해 사용자의 맞춤 서울 코스를 추천하세요.

        [도구 사용 순서]
        1. elastic_search: 사용자의 질문에서 '지역'과 '카테고리' 키워드를 추출해 elastic_search 도구를 사용하세요.
        2. search_with_filtering: 사용자의 쿼리, elastic_search에서 반환한 장소의 uuid 리스트를 모두 사용해 search_with_filtering 도구를 호출하세요.

        - 각 도구의 입력값은 이전 단계의 출력값을 반드시 활용하세요.
        - 도구를 순서대로 모두 사용한 후에만 최종 코스를 추천할 수 있습니다.
        - 장소 데이터에 없는 장소는 상상하지 말고, 반드시 도구 결과만 사용하세요.
        - 따뜻하고 친근한 말투로 대화하세요.

        [코스 생성 규칙]
        - `search_with_filtering` 도구에서 반환된 장소 목록을 분석해야 합니다.
        - 사용자의 요청에 언급된 각 활동(예: 카페, 한식, 박물관)에 대해 가장 적합한 장소를 **하나씩만** 선택하세요.
        - 사용자가 언급한 순서대로 코스를 구성해야 합니다. 예를 들어, "카페 갔다가 밥 먹고 싶어"라고 했다면, 추천된 장소 목록에서 카페 하나와 음식점 하나를 선택하여 "1. OO카페\\n2. XX식당" 순서로 코스를 만들어야 합니다.
        - 최종적으로 구성된 코스에 포함된 장소들의 UUID만 `placeid` 리스트에 담아주세요.
        - 생성된 코스는 `str` 필드에 친절한 소개 문구와 함께 담아주세요.

        [출력 포맷]
        - 모든 답변은 아래 JSON 형식을 반드시 따르세요.
        - `placeid`는 줄바꿈을 사용하지 마세요.
        - `str`은 줄바꿈과 단락을 구분하세요. `\n`을 통해서 줄바꿈을 표시하고, `\n<br>`을 통해서 단락이 바뀜을 표시하세요.
        - `<n>`과 `<br>`은 줄바꿈과 단락을 구분하는 문자입니다. 이외의 문자는 사용하지 마세요.
        - 주소와 장소 사이에는 반드시 줄바꿈으로 구분하세요.
        - 다음 장소의 설명으로 넘어갈 때, 단락을 구분하세요.
        {{
          "placeid": ["uuid1", ...],
          "str": "추천 코스에 대한 친근한 설명과 함께 장소 목록을 제공합니다. 예: 요청하신 종로 데이트 코스입니다.<br>1. 장소명 - 주소<n>설명<br>2. 장소명 - 주소<n>설명"
        }}

        [주의]
        - 반드시 elastic_search → search_with_filtering 순서로 도구를 호출하세요.
        - 도구 호출 없이 임의로 장소를 추천하지 마세요.
        - 최종 응답은 JSON 형식만 사용하고, 다른 텍스트는 추가하지 마세요.
        """
        return ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    def _get_user_memory(self, user_id: str) -> ConversationBufferMemory:
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.user_memories[user_id]
        
    def get_response(self, user_message: str, user_id: str) -> dict:
        memory = self._get_user_memory(user_id)
        
        agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,  # 디버깅을 위한 로그 출력
            handle_parsing_errors=True # 파싱 에러 처리
        )

        try:
            # `user_id`를 `user_uuid`로 전달하여 에이전트 호출
            result = agent_executor.invoke(
                {"input": user_message},
                config={"callbacks": [StdOutCallbackHandler()]}
            )
            
            output_str = result.get("output", "{}")
            try:
                # 에이전트의 출력이 JSON 문자열일 경우 파싱
                response_data = json.loads(output_str)
            except (json.JSONDecodeError, TypeError):
                # JSON 파싱 실패 시, 추천 코스를 담은 텍스트 응답으로 처리
                response_data = {"placeid": [], "str": output_str}

            return response_data

        except Exception as e:
            print(f"에이전트 실행 중 오류 발생: {e}")
            return {"placeid": [], "str": "죄송합니다. 처리 중 오류가 발생했습니다."}

    def clear_session(self, user_id: str):
        if user_id in self.user_memories:
            del self.user_memories[user_id]
            print(f"--- 대화 기록 삭제: {user_id} ---")