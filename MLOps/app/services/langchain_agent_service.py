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
        
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
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
        
        self.tools = [
            self._create_combined_search_tool(),
            self._create_congestion_tool()
        ]
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
                    "area": meta.get("area"),
                    "address": meta.get("address"),
                    "content": meta.get("content", "")
                })
            return essential_data

        return search_course

    def _create_congestion_tool(self):
        @tool
        async def get_congestion_forecast(regions: List[str]) -> Dict[str, Dict[str, str]]:
            """
            주어진 여러 지역의 1시간, 3시간 및 6시간 후 예상 혼잡도를 한 번에 조회하는 도구입니다.
            코스 추천이 완료된 후, 해당 지역들의 전반적인 미래 혼잡도를 사용자에게 알려주기 위해 사용됩니다.
            이 도구는 반드시 `search_course` 도구가 성공적으로 장소 목록을 반환한 후에만 호출되어야 합니다.

            Args:
                regions (List[str]): 혼잡도를 조회할 지역명 리스트 (예: ['발산역', '충정로역']).

            Returns:
                dict: 각 지역별로 1시간, 3시간, 6시간 후 예상 혼잡도 정보를 담은 딕셔너리.
                      (예: {'강남': {'1_hour_forecast': '보통', '3_hour_forecast': '보통', '6_hour_forecast': '붐빔'}, '홍대': {'1_hour_forecast': '여유', '3_hour_forecast': '여유', '6_hour_forecast': '보통'}})
            """
            base_url = os.getenv("EC2_HOST_ML", "http://mlops-backend:8000")
            results = {region: {} for region in regions}

            async with httpx.AsyncClient() as client:
                for hour in [1, 3, 6]:
                    try:
                        # httpx가 지원하고 linter 오류를 해결하는 dict 형식으로 파라미터 구성
                        params = {'hour': str(hour), 'area': regions}
                        
                        url = f"{base_url}/api/crowd"
                        response = await client.get(url, params=params, timeout=10)
                        
                        if response.status_code == 404:
                            print(f"Congestion prediction file not found for hour={hour}, regions={regions}")
                            for region in regions:
                                results[region][f'{hour}_hour_forecast'] = "예측 정보 없음"
                            continue
                        
                        response.raise_for_status()
                        data = response.json()
                        
                        if data.get("success") and data.get("crowdLevel", {}).get("total", 0) > 0:
                            # 응답에서 지역명-혼잡도 맵 생성
                            congestion_map = {row["area_nm"]: row["area_congest_lvl"] for row in data["crowdLevel"]["row"]}
                            for region in regions:
                                results[region][f'{hour}_hour_forecast'] = congestion_map.get(region, "정보 없음")
                        else:
                            for region in regions:
                                results[region][f'{hour}_hour_forecast'] = "정보 없음"

                    except Exception as e:
                        print(f"Congestion API call error (hour={hour}, regions={regions}): {e}")
                        for region in regions:
                            results[region][f'{hour}_hour_forecast'] = "정보를 가져올 수 없습니다"
            
            return results
        
        return get_congestion_forecast

    def _create_prompt_template(self):
        template = """
        # 역할
        너는 '하루'라는 이름의 서울 하루 코스 추천 챗봇이야. 밝고 친근한 말투와 적절한 이모티콘을 사용해. 
        너는 서울 지역에 대해서 매우 잘 알고있고, 서울 외 지역은 추천하지 마.
        추가적으로, 너는 사용자가 원하는 장소들도 찾아줄 수 있어.

        # 출력 형식
        모든 응답은 아래 JSON 형식으로 출력해야 해.  
        - `placeid`: 추천 장소 UUID 목록 (없으면 빈 리스트)  
        - `str`: 사용자에게 보여줄 메시지 (예시 형식 준수)  

        ```json
        {{
        "placeid": ["장소 UUID 목록"],
        "str": "사용자에게 보여줄 메시지"
        }}
        ```
        
        # 도구 사용 규칙
        - 장소 검색은 반드시 지역과 카테고리가 명확할 때만 search_course 도구 호출
        - 장소 또는 카테고리만 입력되면 추천하지 마.
        - 장소 목록이 반환되면, 해당 장소들의 지역(area) 값으로 중복 제거하여 get_congestion_forecast 도구를 1회만 호출
        - 모든 장소 정보는 search_course 결과로만 사용 (사전 지식 사용 금지)
        
        # 툴 호출 흐름 규칙
        - 장소 검색 결과(search_course 도구의 출력)는 반드시 응답에 포함될 장소 후보들의 정보만 사용해야 해.
        - 장소 정보에는 'area' 필드가 반드시 포함되며, 이 값들을 모두 수집한 뒤 **중복을 제거**해 리스트로 만들고, 이를 get_congestion_forecast 도구에 넘겨야 해.
        - get_congestion_forecast 도구는 반드시 한 번만 호출하고, 위에서 정리한 지역 리스트 전체를 한 번에 전달해야 해.
        - 절대로 지역별로 여러 번 나누어 호출하지 말 것.
        - 도구 호출 순서: 먼저 search_course → 그 결과로 area 수집 → 중복 제거 → get_congestion_forecast 한 번 호출

        # 메시지 작성 가이드
        - 첫 문장은 "[지역]에서 [카테고리] 즐기는 코스야!"로 시작
        - 장소 최대 6개 추천
        - <br>, <n> 을 제외한 다른 모든 태그는 절대 사용 금지
        - 장소 설명 형식:
        <br>1. 상호명 - 주소<n>설명<br>2. ...

        - 혼잡도는 마지막에 추가.
        ## 단일 지역:
        <br>✨ [지역] 지역 (포함된 추천 장소) 혼잡도 예보<n>- 1시간 후: ○○<n>- 3시간 후: ○○<n>- 6시간 후: ○○
        ## 다중 지역:
        <br>✨ 지역별 혼잡도 예보<n>📍 지역명 (포함된 추천 장소)<n>- 1시간 후: ○○<n>- 3시간 후: ○○<n>- 6시간 후: ○○ ...

        # 기타 규칙
        - 정보가 부족할 땐 되물어보고 placeid는 빈 리스트로 반환
        - "기다려줘", "내가 찾아볼게" 같은 말은 금지
        - 6곳 이상 요청 시 "먼저 6곳만 추천해줄게!"라고 알려줘
        - 장소 설명은 항상 부드럽고 따뜻하게 재구성해줘
        - 사용자의 요청이 명확하면 즉시 검색 도구를 사용해 코스를 구성하고 최종 응답을 완성해야 해.
        - 중간에 "찾아볼게요", "잠시만요", "기다려줘요" 같은 진행 멘트를 쓰면 안 돼.
        - 항상 완성된 추천 결과처럼 보이도록 구성해. 한 번에 끝내야 해.
        - 코스 생성을 하게 되면, 반드시 순서를 고려하고 사용자가 요청한 카테고리 별로 하나의 장소로만 추천해.
        - 예시:  
        - 사용자 입력: "강남에서 전시, 카페, 한식 가고 싶어"  
        - 추천해야 할 장소: 전시관 1곳, 카페/전통찻집 1곳, 한식 1곳 

        # 예시 매핑
        "강남역에서 파스타 먹고 싶어" → 지역: "강남", 카테고리: "음식점&카페"
        "홍대에서 방탈출" → 지역: "홍대", 카테고리: "레포츠"
        "충정로역에서 카페 갔다가 전시 보는 코스" → 지역: "충정로", 카테고리: "전시관"
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