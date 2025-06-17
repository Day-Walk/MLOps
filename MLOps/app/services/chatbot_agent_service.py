import os
import requests
import json
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
            'elastic_search'를 통해 얻은 uuid_list로 장소들을 1차 필터링한 후,
            사용자의 원래 질문(query)과의 유사도를 기반으로 가장 관련성 높은 장소 목록을 최종적으로 반환합니다.
            - uuid_list: 'elastic_search' 도구에서 반환된 장소 UUID 목록입니다.
            - query: "홍대 분위기 좋은 카페 추천해줘" 와 같이, 사용자가 입력한 원래의 전체 질문입니다. 이 전체 질문을 통해 장소의 분위기나 특징과의 유사성을 파악합니다.
            """
            retriever = self.chroma_db.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 100, # 너무 많은 결과를 반환하지 않도록 k값 조정
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
You have two models available: a "Course Recommendation AI" and a "General Conversation AI". Based on the user's input, determine which model is more appropriate to generate a response.

[Course Recommendation AI]
This AI is activated when the user asks for a place recommendation. It follows a strict, step-by-step process.

#Step 1: Use the `elastic_search` tool.
- Extract the 'region' and 'category' from the user's query to use as arguments for the `elastic_search` tool.
- If the necessary information is not in the query, ask the user for it.

#Step 2: Use the `search_with_filtering` tool.
- Use the `uuid_list` obtained from `elastic_search` and the user's original query as arguments for the `search_with_filtering` tool.

#Step 3: Generate the final course recommendation.
- Synthesize the information from the `search_with_filtering` results to create the final response.
- The `search_with_filtering` tool returns a list of places, where each place consists of `page_content` (a detailed description) and `metadata` (containing `name`, `address`, `uuid`, etc.).
- Use this information to compose a creative and appealing `str1` (overall course description) and `str2` (detailed description of each place).
- The final response MUST be in the following JSON format. Do not add any other text.
{{
  "str1": "This is an overall description of the date course tailored to the user's request. For example, 'A harmonious day course in Hongdae, blending art and cuisine.'",
  "str2": "This provides a detailed description of each recommended place. Please format it as '1. [Place Name]: [Place Description] \\n2. [Place Name]: [Place Description]'. Use a colon (:) after the place name and a newline character (\\n) to separate each place."
}}

[General Conversation AI]
- This AI is activated for all other conversations, such as simple greetings or chit-chat.
- It should respond naturally and kindly in plain text, not in JSON format.
- Maintain a warm and friendly tone (e.g., "Of course! Let's plan a course together slowly ^^").
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
        
        str1 = ""
        str2 = ""
        try:
            # 응답이 JSON 형식일 수 있으므로 파싱 시도
            answer_json = json.loads(final_answer)
            str1 = answer_json.get("str1", "")
            str2 = answer_json.get("str2", "")
        except (json.JSONDecodeError, TypeError):
            # JSON 파싱 실패 시 일반 텍스트 답변으로 간주
            str1 = final_answer

        # 최종 응답 생성. placeid가 비어있더라도 str1, str2는 채워져 있을 수 있음
        return {"str1": str1, "placeid": ordered_uuids, "str2": str2} 