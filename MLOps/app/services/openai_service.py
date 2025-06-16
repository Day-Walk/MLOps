"""
OpenAI API 서비스
실제 GPT API와 연결하여 동기/비동기 응답 생성
"""
from openai import AsyncOpenAI
from typing import AsyncGenerator, Dict, Any

class OpenAIService:
    """OpenAI API 서비스 클래스"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        
        # 데이트 코스 추천을 위한 시스템 프롬프트
        self.system_prompt = """
당신은 데이트 코스를 추천하는 전문가입니다.

역할:
- 사용자의 지역, 분위기, 선호도를 파악해서 데이트 코스를 추천합니다.
- 따뜻하고 친근한 말투로 대화합니다.
- 한 번에 하나씩 질문하며 정보를 수집합니다.

응답 규칙:
1. 충분한 정보가 수집되면 구체적인 장소들로 코스를 추천하세요.
2. 추천할 때는 다음 형식을 사용하세요:

   이런 코스는 어떠세요?

   1. [장소 이름] - [주소/지역]
   - 장소에 대한 간단한 설명
   
   2. [장소 이름] - [주소/지역] 
   - 장소에 대한 간단한 설명

3. 추천 후에는 "다른 코스도 더 보고 싶으신가요?"라고 물어보세요.
"""
    
    async def get_chat_completion(self, message: str) -> str:
        """동기식 채팅 완료"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ OpenAI API 호출 실패: {e}")
            return "죄송합니다. 현재 서비스에 일시적인 문제가 있습니다."
    
    async def get_streaming_completion(self, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """비동기 스트리밍 채팅"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield {
                        "type": "chunk",
                        "content": content
                    }
                
                if chunk.choices[0].finish_reason == "stop":
                    yield {
                        "type": "complete",
                        "content": ""
                    }
                    break
            
        except Exception as e:
            print(f"❌ OpenAI 스트리밍 실패: {e}")
            yield {
                "type": "error",
                "content": f"OpenAI API 오류: {str(e)}"
            }
    
    async def test_connection(self) -> bool:
        """OpenAI API 연결 테스트"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "테스트"}],
                max_tokens=10,
                stream=False
            )
            return True
        except Exception as e:
            print(f"❌ OpenAI 연결 테스트 실패: {e}")
            return False
