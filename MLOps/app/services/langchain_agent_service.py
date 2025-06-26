import os
import json
import httpx
import re
import asyncio
import threading
from urllib.parse import quote
from typing import List, Dict, Any
import functools
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma

load_dotenv()


class LangchainAgentService:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다.")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
        
        # 임베딩 모델 초기화
        self.model_bge = HuggingFaceEmbeddings(
            model_name="upskyy/bge-m3-korean",
            model_kwargs={'device': 'cpu'},
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
        
        self.tools = [self._create_combined_search_tool()]
        self.prompt = self._create_prompt_template()
        self.user_memories = {}
        self._lock = threading.Lock()

    def _create_combined_search_tool(self):
        @tool
        async def search_course(query: str, region: str, categories: List[str]) -> List[Dict[str, Any]]:
            """
            사용자의 쿼리를 기반으로 지역과 카테고리에 맞는 장소를 검색하고,
            상세 쿼리를 사용해 가장 관련성 높은 장소의 상세 정보를 반환하는 통합 검색 도구입니다.

            이 도구는 두 단계로 작동합니다:
            1. '지역(region)'과 '카테고리(categories)'로 장소 UUID 목록을 가져옵니다.
            2. 가져온 UUID 목록과 `query`에서 지역과 카테고리를 제외한 문자열을 사용해 벡터 검색으로 상세 정보를 필터링하고 반환합니다.

            Args:
                query (str): 사용자가 입력한 전체 쿼리 (예: '홍대에서 감성 카페 갔다가 전시 보는 코스 추천')
                region (str): 사용자가 명시한 지역명 (예: '강남', '홍대', '종로3가역')
                categories (list of str): 검색할 장소 카테고리 목록입니다.
                사용 가능한 전체 카테고리: '전시관', '기념관', '전문매장/상가', '5일장', '특산물판매점', '백화점', '상설시장', '문화전수시설', '문화원', '서양식', '건축/조형물', '음식점&카페', '박물관', '컨벤션센터', '역사관광지', '복합 레포츠', '공예/공방', '이색음식점', '영화관', '산업관광지', '중식', '문화시설', '쇼핑', '수상 레포츠', '관광지', '육상 레포츠', '학교', '관광자원', '스키(보드) 렌탈샵', '대형서점', '휴양관광지', '외국문화원', '자연관광지', '레포츠', '한식', '일식', '도서관', '체험관광지', '카페/전통찻집', '면세점', '공연장', '미술관/화랑'

            Returns:
                list: 필터링된 장소 정보 리스트. 각 장소는 상세 정보를 담은 딕셔너리입니다.
                      (예: [{'name': 'OO카페', 'address': '서울...', ...}, ...])
            """
            # 1. Elastic Search to get UUIDs
            if not region or not categories:
                return []

            print(f"--- 엘라스틱 검색 (카테고리별 호출): region={region}, categories={tuple(sorted(categories))} ---")

            region_encoded = quote(region)
            all_place_uuids = set()

            async with httpx.AsyncClient() as client:
                tasks = []
                for category in categories:
                    category_encoded = quote(category)
                    url = f"http://15.164.50.188:9201/api/place/search/llm-tool?region={region_encoded}&categories={category_encoded}"
                    tasks.append(client.get(url, timeout=10))

                api_responses = await asyncio.gather(*tasks, return_exceptions=True)

                for response in api_responses:
                    if isinstance(response, httpx.Response):
                        try:
                            response.raise_for_status()
                            response_json = response.json()
                            uuids = response_json.get('uuids', [])
                            all_place_uuids.update(uuids)
                        except (httpx.HTTPStatusError, json.JSONDecodeError) as e:
                            print(f"API 응답 처리 중 오류: {e}")
                    elif isinstance(response, Exception):
                        print(f"엘라스틱 검색 API 호출 오류: {response}")
            
            place_uuids = list(all_place_uuids)
            print(f"--- 엘라스틱 검색 결과 (통합): {len(place_uuids)}개 장소 ---")

            if not place_uuids:
                return []

            # 2. Embedding search with filtering
            print(f"--- 필터링 및 임베딩 검색: query='{query}', UUIDs={len(place_uuids)}개 ---")

            retriever = self.chroma_bge.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 15,
                    "filter": {"uuid": {"$in": place_uuids}}
                }
            )
            documents = await retriever.ainvoke(query)
            essential_data = []
            for doc in documents:
                meta = doc.metadata
                essential_data.append({
                    "uuid": meta.get("uuid"),
                    "name": meta.get("name"),
                    "address": meta.get("address"),
                    "content": meta.get("content", "")
                })
            return essential_data

        return search_course

    def _create_prompt_template(self):
        template = """
        1. 페르소나 및 역할
        너의 이름은 "하루", 서울 지리에 능통한 활기찬 챗봇이야.
        너의 목표는 사용자가 최고의 하루를 보낼 수 있도록 최적의 코스를 추천하는 것이야.
        말투는 항상 긍정적이고 다정하며, "내가 딱 좋은데 찾아놨지! 😉"처럼 이모티콘을 적절히 사용해.
        
        2. 핵심 임무: 지정 JSON 형식으로만 출력
        너의 모든 응답은 반드시 아래의 유효한 JSON 형식이어야 해. 이건 가장 중요한 기술적 제약사항이야.
        
        ```json
        {{
        "placeid": ["장소 UUID 목록"],
        "str": "사용자에게 보여줄 메시지"
        }}
        ```
        3. 작동 원칙 및 도구(search_course) 사용 규칙
        실행 조건: 지역과 카테고리가 대화에서 모두 명확하게 확정되었을 때만 도구를 사용해.
        정보 부족 시: 지역이나 카테고리 정보가 부족하면 "오, 좋아! 혹시 어느 동네 쪽으로 알아보고 있어?" 와 같이 질문해야 해. 이때 placeid는 반드시 빈 리스트 []여야 해.
        카테고리 매핑: 사용자의 자연어(예: 파스타, 방탈출, 옷 구경)를 지정된 카테고리(예: 서양식, 레포츠, 쇼핑)로 변환하여 검색해야 해.
        데이터 출처 (Strict RAG): 장소에 대한 모든 정보(이름, 주소, 설명)는 오직 search_course 도구로 검색된 결과만 사용해야 해. 너의 사전 지식을 절대 사용하면 안 돼.
        
        4. str 필드 작성 가이드
        코스 요약: 첫 문장은 항상 "[지역]에서 [카테고리] 즐기는 코스!"와 같이 요약으로 시작해.
        코스 추천: 특별한 요청이 없으면 카테고리당 1곳을 추천하고, 코스 전체에 포함되는 장소는 최대 6개로 제한해.
        장소 추천: 특정한 장소만 추천해줄 때는 최대 6개로 제한해.
        내용 형식:
        각 장소는 반드시 번호. 상호명 - 주소<n>설명 순서로 작성해.
        장소와 장소 사이는 <br> 하나로만 구분해야 해.
        (예시: <br>1. 상호명 - 주소<n>설명<br>2. 상호명 - 주소<n>설명<br>3. 상호명 - 주소<n>설명)
        
        5. 지역 및 카테고리 매핑 예시
        - 사용자의 자연어 요청을 도구에서 사용할 수 있도록 지역은 이해하고, 카테고리는 사용 가능한 전체 카테고리로 변환해.
        - 사용자 입력: "강남에서 파스타 먹고 싶어" -> 매칭 지역: "강남", 매칭 카테고리: "서양식"
        - 사용자 입력: "홍대에서 방탈출 할만한 곳 있어?" -> 매칭 지역: "홍대", 매칭 카테고리: "레포츠"
        - 사용자 입력: "성수동에서 케이크 맛있는 데" -> 매칭 지역: "성수동", 매칭 카테고리: "카페/전통찻집"
        - 사용자 입력: "연남동에서 옷 구경하고 싶어" -> 매칭 지역: "연남동", 매칭 카테고리: "쇼핑"
        
        6. 추가 규칙
        - 사용자가 요구하는 장소의 갯수가 6개 이상일 때는 "최대 6개까지만 추천해줄게!" 라는 말을 포함해서 한 번의 답변으로 장소를 추천해줘.
        - 사용자가 특정 장소에 대해서 고르거나 수정을 요구하면 그 장소에 맞는 placeid와 장소명, 주소, 설명을 모두 기억해.
        - 문서를 기반으로 장소 설명을 할 때, "이런 장소입니다." 처럼 딱딱한 말투는 절대로 사용하지마. 친근하고 이모티콘을 사용해서 재구성해.
        """
        return ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    def _get_user_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        with self._lock:
            if user_id not in self.user_memories:
                self.user_memories[user_id] = ConversationBufferWindowMemory(
                    k=6,
                    memory_key="chat_history",
                    return_messages=True,
                )
            return self.user_memories[user_id]
        
    async def get_response(self, user_message: str, user_id: str) -> dict:
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
            result = await agent_executor.ainvoke(
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