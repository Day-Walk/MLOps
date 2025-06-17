import os
import requests
from urllib.parse import quote
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage

class ChatbotAgentService:
    def __init__(self, openai_api_key: str, chroma_db_path: str = "data/chroma_db_bge"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="upskyy/bge-m3-korean",
            model_kwargs={'device': 'cpu'},  # API 서버 환경에 맞게 CPU로 변경
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.chroma_db = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=self.embedding_function
        )
        
        self.tools = self._get_tools()
        self.agent_executors = {}  # 세션별 AgentExecutor 저장

    def _get_tools(self):
        # 노트북에서 정의된 도구들을 클래스 메소드로 변환
        
        @tool
        def elastic_search(region: str, categories: list[str]):
            """
            사용자가 검색을 원하는 '지역(region)'과 '카테고리(categories)'를 인자로 받아,
            해당 조건에 맞는 장소들의 UUID 리스트를 반환합니다.
            - region : 사용자가 명시한 지역 이름 (예 : "강남", "홍대", "성수동", "이태원로", "잠실역" 등).
            - categories : 사용자가 원하는 활동이나 장소 유형의 예시 중에서 최소 1개 최대 3개를 리스트로 받습니다.
            - (예시 : '전시관', '기념관', '전문매장/상가', '5일장', '특산물판매점', '백화점', '상설시장', '문화전수시설', '문화원', '서양식', '건축/조형물', '음식점&카페', '박물관', '컨벤션센터', '역사관광지', '복합 레포츠', '공예/공방', '이색음식점', '영화관', '산업관광지', '중식', '문화시설', '쇼핑', '수상 레포츠', '관광지', '육상 레포츠', '학교', '관광자원', '스키(보드) 렌탈샵', '대형서점', '휴양관광지', '외국문화원', '자연관광지', '레포츠', '한식', '일식', '도서관', '체험관광지', '카페/전통찻집', '면세점', '공연장', '미술관/화랑')

            - 정확한 카테고리가 들어오지 않더라도 유사한 카테고리를 추출하여 인자로 받아주세요.
            """
            if not region or not categories:
                return "지역과 카테고리를 정확히 입력해주세요."

            region_encoded = quote(region)
            category_encoded = '&'.join([f"categories={quote(cat)}" for cat in categories])

            url = f"http://15.164.50.188:9201/api/place/search/llm-tool?region={region_encoded}&{category_encoded}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                response_json = response.json()
                uuid_list = response_json.get('uuids', [])
                return uuid_list
            except requests.exceptions.RequestException as e:
                return f"Elasticsearch API 호출 중 오류 발생: {e}"

        @tool
        def search_with_filtering(uuid_list: list[str], query: str):
            """
            uuid_list로 사전 필터링한 후, 사용자 쿼리와의 유사도를 기반으로 최종 장소 목록을 검색합니다.
            이 도구는 elastic_search를 통해 얻은 uuid_list를 사용해 더 정확한 결과를 찾을 때 사용됩니다.
            - uuid_list: 'elastic_search' 도구에서 반환된 장소 UUID 목록입니다.
            - query: 사용자의 원래 질문 또는 의도입니다. (예: "분위기 좋은 카페 가고 싶어")
            """
            retriever = self.chroma_db.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 10, # 너무 많은 결과를 반환하지 않도록 k값 조정
                    "filter": {
                        "uuid": {"$in": uuid_list}
                    }
                }
            )
            place_list = retriever.invoke(query)
            # Document 객체를 직렬화 가능한 형태로 변환
            return [doc.dict() for doc in place_list]

        return [elastic_search, search_with_filtering]

    def _create_agent_executor(self, memory):
        prompt_template = """
당신은 사용자에게 맞춤형 데이트 코스를 추천하는 똑똑하고 친절한 챗봇 '데이워커'입니다.

【역할 및 지침】
- 당신의 주된 목표는 사용자와 대화하여 원하는 '지역'과 '카테고리'를 파악하고, 이를 바탕으로 장소를 추천하는 것입니다.
- 사용자가 처음 장소를 추천해달라고 하면, 먼저 `elastic_search` 도구를 사용하여 장소 후보 목록을 가져와야 합니다.
- `elastic_search`를 통해 얻은 장소 목록(uuids)이 너무 많거나 사용자의 의도가 더 구체적이라면, `search_with_filtering` 도구를 사용하여 결과를 더 좁힐 수 있습니다.
- 항상 제공된 도구를 사용하여 정보를 검색해야 하며, 절대로 임의의 장소를 만들거나 당신의 지식에 기반해 답변해서는 안 됩니다.
- 최종 추천은 `search_with_filtering` 도구로부터 받은 장소 목록을 바탕으로, 아래 '응답 형식'에 맞게 제공해주세요.
- 사용자의 질문이 불분명하면, "어떤 지역을 원하세요?" 또는 "무엇을 하고 싶으신가요?" 와 같이 명확한 질문을 통해 필요한 정보를 얻으세요.

【말투】
- 항상 따뜻하고 친근한 말투를 사용하세요. (예: "좋아요! 천천히 같이 코스를 짜볼게요 ^^", "분위기 좋은 곳 위주로 찾아볼게요.")

【응답 형식】
- 충분한 정보가 수집되어 코스를 추천할 때, 아래 형식을 따라주세요.

    • 이런 코스는 어떠세요?

    1. [장소 이름] - [주소]
    - 장소 설명
    2. [장소 이름] - [주소]
    - 장소 설명
"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=prompt_template),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            return_intermediate_steps=True  # 중간 단계 반환 설정
        )
        return agent_executor

    def _get_or_create_agent_executor(self, session_id: str):
        if session_id not in self.agent_executors:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.agent_executors[session_id] = self._create_agent_executor(memory)
        return self.agent_executors[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.agent_executors:
            del self.agent_executors[session_id]
            return True
        return False

    def get_response(self, user_message: str, session_id: str):
        executor = self._get_or_create_agent_executor(session_id)
        response = executor.invoke({"input": user_message})
        
        final_answer = response.get("output", "죄송합니다, 답변을 생성하지 못했습니다.")
        
        # 중간 단계에서 장소 UUID 목록을 순서대로 추출
        ordered_uuids = []
        intermediate_steps = response.get('intermediate_steps', [])
        # 에이전트가 마지막으로 호출한 필터링 도구의 결과를 사용
        for action, tool_output in reversed(intermediate_steps):
            if action.tool == 'search_with_filtering' and isinstance(tool_output, list):
                for doc in tool_output:
                    # Document 직렬화 결과에서 메타데이터와 uuid 추출
                    if 'metadata' in doc and 'uuid' in doc['metadata']:
                        ordered_uuids.append(doc['metadata']['uuid'])
                break # 마지막 결과만 사용하므로 반복 중단
        
        return {"final_answer": final_answer, "place_uuids": ordered_uuids} 