import os
import requests
import json
from urllib.parse import quote
from typing import Union, List, Dict
import chromadb

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# New Prompt Template for OpenAI Functions Agent
TEMPLATE = """
당신은 사용자에게 맞춤형 데이트 코스를 추천하는 똑똑하고 친절한 챗봇 '데이워커'입니다.

【역할 및 지침】
- 당신의 주된 목표는 사용자와 대화하여 원하는 '지역'과 '카테고리'를 파악하고, 이를 바탕으로 장소를 추천하는 것입니다.
- 사용자가 처음 장소를 추천해달라고 하면, 먼저 `elastic_search` 도구를 사용하여 장소 후보 목록을 가져와야 합니다.
- `elastic_search`를 통해 얻은 장소 목록(uuids)이 너무 많거나 사용자의 의도가 더 구체적이라면, `search_with_filtering` 도구를 사용하여 결과를 더 좁힐 수 있습니다.
- `search_with_filtering` 도구를 사용할 때는 `query` 인자에 반드시 **사용자의 원래 입력 메시지 전체**를 사용하여 의미적으로 유사한 장소를 찾아야 합니다.
- 항상 제공된 도구를 사용하여 정보를 검색해야 하며, 절대로 임의의 장소를 만들거나 당신의 지식에 기반해 답변해서는 안 됩니다.
- `search_with_filtering`에서 반환된 결과에서 'metadata'의 'uuid'는 'placeid'로, 'page_content'는 'str2'의 각 장소 설명으로 사용하세요. 전체적인 설명은 'str1'에 담아주세요.
- 사용자의 질문이 불분명하면, "어떤 지역을 원하세요?" 또는 "무엇을 하고 싶으신가요?" 와 같이 명확한 질문을 통해 필요한 정보를 얻으세요.

【말투】
- 항상 따뜻하고 친근한 말투를 사용하세요. (예: "좋아요! 천천히 같이 코스를 짜볼게요 ^^", "분위기 좋은 곳 위주로 찾아볼게요.")

【최종 응답 형식】
- 충분한 정보가 수집되어 최종적으로 코스를 추천할 때는, 반드시 아래와 같은 JSON 형식으로만 응답해야 합니다. 다른 말은 섞지 마세요.
{{
    "str1" : "코스에 대한 전체적인 설명",
    "placeid" : ["placeid1", "placeid2", ...],
    "str2" : "각 장소에 대한 설명"
}}
"""

class ChatbotAgentService:
    def __init__(self, chroma_db_path: str = "/app/data/chroma_db_bge"):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="upskyy/bge-m3-korean",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        client_settings = chromadb.Settings(
            is_persistent=True,
            persist_directory=chroma_db_path,
        )
        self.chroma_db = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=self.embedding_function,
            client_settings=client_settings
        )
        
        self.tools = self._get_tools()
        self.agent_executors = {}

    def _get_tools(self):
        @tool
        def elastic_search(region: str, categories: List[str]) -> List[str]:
            """
            사용자가 검색을 원하는 '지역(region)'과 '카테고리(categories)'를 인자로 받아,
            해당 조건에 맞는 장소들의 UUID 리스트를 반환합니다.
            - region : 사용자가 명시한 지역 이름 (예 : "강남", "홍대", "성수동", "이태원로", "잠실역" 등).
            - categories : 사용자가 원하는 활동이나 장소 유형의 예시 중에서 최소 1개 최대 3개를 리스트로 받습니다.
            - (예시 : '전시관', '기념관', '전문매장/상가', '5일장', '특산물판매점', '백화점', '상설시장', '문화전수시설', '문화원', '서양식', '건축/조형물', '음식점&카페', '박물관', '컨벤션센터', '역사관광지', '복합 레포츠', '공예/공방', '이색음식점', '영화관', '산업관광지', '중식', '문화시설', '쇼핑', '수상 레포츠', '관광지', '육상 레포츠', '학교', '관광자원', '스키(보드) 렌탈샵', '대형서점', '휴양관광지', '외국문화원', '자연관광지', '레포츠', '한식', '일식', '도서관', '체험관광지', '카페/전통찻집', '면세점', '공연장', '미술관/화랑')

            - 정확한 카테고리가 들어오지 않더라도 유사한 카테고리를 추출하여 인자로 받아주세요.
            """
            if not region or not categories:
                return "Region and categories are required."

            region_encoded = quote(region)
            category_encoded = '&'.join([f"categories={quote(cat)}" for cat in categories])
            url = f"http://15.164.50.188:9201/api/place/search/llm-tool?region={region_encoded}&{category_encoded}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                return response.json().get('uuids', [])
            except requests.exceptions.RequestException as e:
                return [f"API call error: {e}"]
            except json.JSONDecodeError:
                return ["API response is not a valid JSON."]

        @tool
        def search_with_filtering(uuid_list: list[str], query: str) -> List[dict]:
            """
            uuid_list로 사전 필터링한 후, 사용자 쿼리와의 유사도를 기반으로 최종 장소 목록을 검색합니다.
            이 도구는 elastic_search를 통해 얻은 uuid_list를 사용해 더 정확한 결과를 찾을 때 사용됩니다.
            - uuid_list: 'elastic_search' 도구에서 반환된 장소 UUID 목록입니다.
            - query: 사용자의 원래 질문 또는 의도입니다. (예: "분위기 좋은 카페 가고 싶어")
            """
            retriever = self.chroma_db.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k" : 100,
                    "filter" : {
                        "uuid" : {"$in": uuid_list}
                    }
                }
            )
            docs = retriever.invoke(query)
            return [doc.dict() for doc in docs]

        return [elastic_search, search_with_filtering]

    def get_agent_executor(self, session_id: str) -> AgentExecutor:
        if session_id not in self.agent_executors:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            prompt = ChatPromptTemplate.from_messages([
                ("system", TEMPLATE),
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
                handle_parsing_errors=True
            )
            self.agent_executors[session_id] = agent_executor
        return self.agent_executors[session_id]
        
    def get_response(self, user_message: str, session_id: str):
        agent_executor = self.get_agent_executor(session_id)
        response = agent_executor.invoke({"input": user_message})
        output = response.get("output", "죄송합니다. 답변을 생성하지 못했습니다.")
        
        # LLM이 JSON 형식으로 응답하지 않았을 경우 처리
        try:
            # The output from the agent should be a string that is a valid JSON.
            # We parse it to ensure it's a JSON object before returning.
            if isinstance(output, str):
                return json.loads(output)
            return output
        except json.JSONDecodeError:
            # 만약 JSON 파싱에 실패하면, LLM이 유효한 JSON을 반환하도록 다시 시도하거나
            # 사용자에게 오류를 알리는 메시지를 반환할 수 있습니다.
            # 여기서는 간단히 오류 메시지를 포함한 딕셔너리를 반환합니다.
            return {
                "error": "Failed to parse LLM response as JSON.",
                "response": output
            }