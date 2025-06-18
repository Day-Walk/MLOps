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


# --- Tool Definitions ---

# @tool
# def get_user_info(user_uuid: str) -> dict:
#     """
#     사용자 정보 조회 도구. user_uuid로 데이터베이스에서 사용자 정보(나이, 성별, 선호 카테고리)를 조회합니다.
#     Args:
#         user_uuid (str): 사용자 고유 식별자.
#     Returns:
#         dict: 사용자 정보 {'age': str, 'gender': str, 'like': list[str]}.
#     """
#     # 실제 애플리케이션에서는 이 부분에서 데이터베이스를 조회해야 합니다.
#     # 노트북의 예시처럼 임시 데이터를 사용합니다.
#     print(f"--- 사용자 정보 조회: {user_uuid} ---")
#     # 예시: 특정 사용자에 대해 다른 정보 반환
#     if user_uuid == "some_test_user_id":
#         return {
#             "age": "20",
#             "gender": "여성",
#             "like": ['조용한', '깔끔한']
#         }
#     return {
#       "age" : "20",
#       "gender" : "남성",
#       "like" : ['카페', '조용한', '깔끔한']
#     }

@tool
def elastic_search(query: dict) -> list:
    """
    엘라스틱 검색 도구. 지역(region)과 카테고리(categories)로 장소 UUID 리스트를 반환합니다.
    Args:
        query (dict): 검색 조건 {'region': str, 'categories': list[str]}.
    Returns:
        list: 장소 UUID 리스트.
    """
    region = query.get('region')
    categories = query.get('categories')
    if not region or not categories:
        return []

    print(f"--- 엘라스틱 검색: region={region}, categories={categories} ---")

    region_encoded = quote(region)
    category_encoded = '&'.join([f"categories={quote(cat)}" for cat in categories])
    # 환경변수나 설정 파일에서 URL을 관리하는 것이 좋습니다.
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

# --- Langchain Agent Service Class ---

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
        chroma_db_path = "data/chroma_db_bge"
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
        def search_with_filtering(data: dict) -> List[Dict[str, Any]]:
            """
            임베딩 검색 도구. UUID 리스트와 사용자 쿼리문으로 메타데이터 필터링 및 임베딩 유사도 검색을 수행합니다.
            Args:
                data (dict): 검색 데이터 {'query': str, 'place_uuids': list[str]}.
            Returns:
                list: 필터링된 장소 정보 딕셔너리 리스트.
            """
            query = data.get('query')
            place_uuids = data.get('place_uuids')
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
        # 노트북의 `template_third` 프롬프트
        template = """
        당신은 데이트 코스 추천 챗봇입니다.

        아래의 도구들을 반드시 순서대로 사용해 사용자의 맞춤 데이트 코스를 추천하세요.

        [도구 사용 순서]
        1. elastic_search: 사용자의 질문에서 '지역'과 '카테고리' 키워드를 추출해 elastic_search 도구를 사용하세요.
        2. search_with_filtering: 사용자의 쿼리, elastic_search에서 반환한 장소의 uuid 리스트를 모두 사용해 search_with_filtering 도구를 호출하세요.

        - 각 도구의 입력값은 이전 단계의 출력값을 반드시 활용하세요.
        - 도구를 순서대로 모두 사용한 후에만 최종 코스를 추천할 수 있습니다.
        - 장소 데이터에 없는 장소는 상상하지 말고, 반드시 도구 결과만 사용하세요.
        - 따뜻하고 친근한 말투로 대화하세요.

        [출력 포맷]
        - 모든 답변은 아래 JSON 형식을 반드시 따르세요.
        {{
          "place_uuids": ["uuid1", ...],
          "course": "추천 코스에 대한 친근한 설명과 함께 장소 목록을 제공합니다. 예: 요청하신 종로 데이트 코스입니다. 1. 장소명 - 주소 - 설명\\n2. 장소명 - 주소 - 설명"
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
            verbose=True  # 디버깅을 위한 로그 출력
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
                response_data = {"place_uuids": [], "course": output_str}

            return response_data

        except Exception as e:
            print(f"에이전트 실행 중 오류 발생: {e}")
            return {"place_uuids": [], "course": "죄송합니다. 처리 중 오류가 발생했습니다."}

    def clear_session(self, user_id: str):
        if user_id in self.user_memories:
            del self.user_memories[user_id]
            print(f"--- 대화 기록 삭제: {user_id} ---")