import os
import json
import requests
import re
from urllib.parse import quote
from typing import List, Dict, Any
import functools
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

load_dotenv()

@functools.lru_cache(maxsize=100)
def _elastic_search_cached(region: str, categories: tuple) -> list:
    """Cached implementation for Elasticsearch search."""
    if not region or not categories:
        return []

    print(f"--- (Cache Miss) 엘라스틱 검색: region={region}, categories={categories} ---")

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

@tool
def elastic_search(region: str, categories: List[str]) -> list:
    """
    엘라스틱 검색 도구

    사용자가 명시한 '지역(region)'과 '카테고리(categories)' 목록을 받아, 해당 조건에 부합하는 장소들의 UUID 리스트를 반환합니다.
    이 도구는 카테고리를 추론하지 않으므로, 반드시 명확한 카테고리 목록을 전달해야 합니다.

    Args:
        region (str): 사용자가 명시한 지역명 (예: '강남', '홍대', '성수동', '이태원로', '잠실역')
        categories (list of str): 검색할 장소 카테고리 목록입니다. (예: ['한식', '카페/전통찻집']). 
        사용 가능한 전체 카테고리: '전시관', '기념관', '전문매장/상가', '5일장', '특산물판매점', '백화점', '상설시장', '문화전수시설', '문화원', '서양식', '건축/조형물', '음식점&카페', '박물관', '컨벤션센터', '역사관광지', '복합 레포츠', '공예/공방', '이색음식점', '영화관', '산업관광지', '중식', '문화시설', '쇼핑', '수상 레포츠', '관광지', '육상 레포츠', '학교', '관광자원', '스키(보드) 렌탈샵', '대형서점', '휴양관광지', '외국문화원', '자연관광지', '레포츠', '한식', '일식', '도서관', '체험관광지', '카페/전통찻집', '면세점', '공연장', '미술관/화랑'

    Returns:
        list: 조건에 부합하는 장소의 UUID 리스트 (예: ['uuid1', 'uuid2', ...])

    Example:
        >>> elastic_search('종로', ['음식점&카페', '문화시설'])
        ['a1b2c3', 'd4e5f6']
    """
    return _elastic_search_cached(region, tuple(sorted(categories)))

class LangchainAgentService:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다.")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        set_llm_cache(InMemoryCache())
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 임베딩 모델 초기화
        self.model_bge = HuggingFaceEmbeddings(
            model_name="upskyy/bge-m3-korean",
            model_kwargs={'device': 'cpu'},  # 'cuda' 사용 가능 시 변경
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # ChromaDB 로드
        chroma_db_path = os.getenv("VECTORDB_PATH")
        if not chroma_db_path or not os.path.exists(chroma_db_path):
            raise FileNotFoundError(f"ChromaDB 경로를 찾을 수 없습니다: {chroma_db_path}")
        self.chroma_bge = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=self.model_bge
        )
        print("--- ChromaDB 로드 완료 ---")
        
        self.tools = [elastic_search, self._create_search_with_filtering_tool()]
        self.prompt = self._create_prompt_template()
        self.user_memories = {}

    def get_cache_info(self):
        """`elastic_search` 도구의 캐시 정보를 반환합니다."""
        info = _elastic_search_cached.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize,
            "current_size": info.currsize
        }

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
        당신은 서울 여행 코스 추천 챗봇입니다. 당신의 유일하고 가장 중요한 임무는 사용자와의 대화 후, 아래 [JSON 출력 형식]에 맞는 유효한 JSON 객체만을 출력하는 것입니다. **어떤 경우에도 JSON 형식이 아닌 텍스트를 출력해서는 안 됩니다.**

        **중요한 제약사항**: 장소에 대한 모든 정보는 반드시 제공된 도구(elastic_search, search_with_filtering)를 통해 검색된 데이터베이스 결과만을 사용해야 합니다. 당신의 사전 지식으로 장소 정보를 추가하거나 보완해서는 안 됩니다.

        [JSON 출력 형식]
        - 모든 응답은 반드시 아래 형식의 JSON 객체여야 합니다.
        ```json
        {{
          "placeid": ["장소 UUID 목록"],
          "str": "사용자에게 보여줄 메시지"
        }}
        ```

        [응답 생성 규칙]
        - 당신은 아래의 규칙에 따라 'str' 필드에 들어갈 메시지를 결정하고, 그 외의 모든 경우에는 추천 장소 목록을 생성합니다.
        - **1. 정보 부족 시 질문**:
            *   대화 전체에서 **'지역'** 정보가 확인되지 않으면, `str` 필드에 어느 지역을 원하는지 5개 이상의 예시를 포함해 물어보는 문구를 담아 JSON을 출력합니다.
            *   '지역'은 있으나 **'할 일'** 정보가 부족하거나 모호하다면, `str` 필드에 어떤 활동을 원하는지 5개 이상의 예시를 포함해 물어보는 문구를 담아 JSON을 출력합니다.
            *   이 경우, `placeid`는 항상 빈 리스트 `[]`입니다.
        - **2. 검색 실패 시 대안 제시**:
            *   도구 검색 결과, 요청한 '지역'에 맞는 장소가 없다면 `str` 필드에 "아쉽게도 요청하신 지역에는 맞는 장소가 없네요. 혹시 강남이나 홍대 같은 다른 인기 지역은 어떠세요?"와 같이 구체적인 대안을 제시하는 질문을 담아 JSON을 출력합니다.
            *   이 경우, `placeid`는 항상 빈 리스트 `[]`입니다.
        - **3. 성공 시 코스 추천**:
            *   도구 검색에 성공하면, `placeid`에는 추천 장소들의 UUID를, `str` 필드에는 [str 필드 작성 규칙]에 따라 생성된 추천 코스 설명을 담아 JSON을 출력합니다.
            *   사용자가 할 일에 대해서, 각각 하나의 장소만을 추천해서 코스를 구성합니다.
            *   반드시 `placeid` 배열에 포함된 `UUID`와 `str` 필드에서 설명하는 장소가 정확히 일치해야 합니다.
            *   예를 들어서, 사용자가 카페를 원하면 카페 카테고리에서 하나를 추천하고, 전시관을 원하면 전시관 카테고리에서 하나를 추천합니다.
        - **4. 코스 수정 요청**:
            *   사용자가 코스 수정을 요청하면 장소 후보를 몇 군데 제시한 후 고르도록 하세요.
            *   이 경우, `placeid`는 항상 빈 리스트 `[]`입니다.

        [도구 사용 규칙]
        - **'지역'과 '할 일'이 대화 전체를 통해 명확하게 확정되었을 때만 도구를 사용합니다.**
        - 사용자가 이전에 말한 '할 일'을 기억했다가, 새로운 지역을 말하면 해당 지역에서 이전에 원했던 '할 일'을 찾아야 합니다.

        [str 필드 작성 규칙 - 중요: RAG 데이터만 사용]
        - **절대적 제약사항**: 장소에 대한 모든 정보(이름, 주소, 설명 등)는 반드시 도구 검색을 통해 얻은 데이터베이스 정보만을 사용해야 합니다. 당신의 사전 지식이나 추측으로 장소 정보를 보완하거나 추가해서는 안 됩니다.
        - 검색된 데이터에 없는 정보는 절대 언급하지 마세요.
        - 항상 부드럽고 친근한 말투로 작성합니다.
        - 코스 추천의 시작은 즐거운 소개 문구로 시작합니다.
        - 각 장소 정보는 반드시 search_with_filtering 도구로 검색된 결과에서만 가져와서 이름 - 주소 - 설명 형식으로 작성합니다.
        - 장소는 항상 번호로 구분하세요. 이 순서는 사용자가 원하는 순서를 따라야 합니다. (예시: 1. 상호명 - 주소<n>장소에 대한 설명)
        - 줄 바꿈은 `<n>`으로 표시하고, 단락 구분은 반드시 `<br>` 하나만 사용합니다.
        - **중요**: `<br>` 태그는 절대로 연속으로 사용하지 마세요. `<br><br>`가 아닌 `<br>` 하나만 사용해야 합니다.
        - 주소와 설명 사이는 `<n>`으로 구분하고, 장소와 장소 사이는 반드시 `<br>` 하나로만 구분합니다.
        - `<n>`과 `<br>` 외의 마크다운은 사용하지 않습니다.
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
            result = agent_executor.invoke(
                {"input": user_message},
                config={"callbacks": [StdOutCallbackHandler()]}
            )
            
            output_str = result.get("output", "{}")
            
            try:
                # 정규식을 사용하여 응답에서 JSON 객체만 추출
                match = re.search(r'\{.*\}', output_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    response_data = json.loads(json_str)
                else:
                    # JSON 객체를 찾지 못한 경우, 전체 문자열을 응답으로 처리
                    response_data = {"placeid": [], "str": "죄송합니다. 처리 중 오류가 발생했습니다."}
            except (json.JSONDecodeError, TypeError):
                # JSON 파싱 실패 시, 추천 코스를 담은 텍스트 응답으로 처리
                response_data = {"placeid": [], "str": "죄송합니다. 처리 중 오류가 발생했습니다."}

            return response_data

        except Exception as e:
            print(f"에이전트 실행 중 오류 발생: {e}")
            return {"placeid": [], "str": "죄송합니다. 처리 중 오류가 발생했습니다."}

    def clear_session(self, user_id: str):
        if user_id in self.user_memories:
            del self.user_memories[user_id]
            print(f"--- 대화 기록 삭제: {user_id} ---")