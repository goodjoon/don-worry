from typing import Dict, Any, Optional, Union, List
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from PIL import Image
import io
from common.llm_factory import LLMFactory
from common.agent.agent_tools import AgentTools


class AgentManager:
    """Agent들을 관리하는 클래스"""
    
    def __init__(self, embedding_model: str = "bge-m3", llm_model: str = "gpt-4.1-mini"):
        """
        AgentManager 초기화
        
        Args:
            embedding_model: 임베딩 모델명
            llm_model: LLM 모델명
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm = LLMFactory.create_llm(model_name=llm_model, temperature=0.0)
        self.agent_tools = AgentTools(embedding_model)
        self.agents = {}
        self.supervisor = None
        
        # Agent들 생성
        self._create_agents()
        self._create_supervisor()
    
    def _create_agents(self):
        """개별 Agent들 생성"""
        # 국토종합계획 검색 Agent
        self.agents["country_plan_search_agent"] = create_react_agent(
            model=self.llm,
            tools=[self.agent_tools.get_country_plan_search_tool()],
            name="country_plan_search_agent",
            prompt="""
                당신은 국토종합계획에 대한 지식을 가진 전문가입니다.
                사용자의 질문에 대해 국토종합계획 문서를 검색하여 관련된 정보를 찾아 답변해주세요.
                
                도구에 사용자의 질의를 전달할 때 아래 사항을 참고하여 질의를 확장해주세요.
                
                **확장 시 고려사항:**
                - 국가 차원의 정책 및 계획 용어 포함
                - 국토개발, 광역계획, 인프라 구축 관련 키워드 추가
                - 장기 국가발전 전략과 연관된 표현 포함
                - 지역균형발전, 국토공간구조 등 국토계획 전문용어 활용

                **주요 확장 키워드:**
                - 국토종합계획, 광역발전계획, 국가균형발전
                - 교통망 구축, 인프라 개발, 지역거점 개발
                - 산업단지, 혁신도시, 경제권역 개발
                - 국토공간구조, 도시체계, 정주여건
            """
        )
        
        # 시도군 기본계획 검색 Agent
        self.agents["city_plan_search_agent"] = create_react_agent(
            model=self.llm,
            tools=[self.agent_tools.get_city_plan_search_tool()],
            name="city_plan_search_agent",
            prompt="""
                당신은 지역 투자를 위한 시도군 단위 개발 기본계획에 대한 지식을 가진 전문가입니다.
                사용자의 질문에 대해 시도군 기본계획 문서를 검색하여 관련된 정보를 찾아 답변해주세요.
                
                사용자의 질의를 시도군 기본계획 문서 검색에 최적화된 형태로 확장해주세요.
                **확장 시 고려사항:**
                - 지역 단위 개발계획 및 도시계획 용어 포함
                - 구체적인 지역 개발사업과 연관된 키워드 추가
                - 주거, 상업, 산업 등 토지이용계획 관련 표현 포함
                - 지역 특성 및 발전 방향과 관련된 용어 활용

                **주요 확장 키워드:**
                - 도시기본계획, 도시관리계획, 지구단위계획
                - 토지이용계획, 주거지 개발, 상업지역 조성
                - 교통계획, 공원녹지, 생활권 계획
                - 재개발, 재건축, 신도시 개발, 택지개발
            """
        )
        
        # 웹 검색 Agent
        self.agents["web_search_agent"] = create_react_agent(
            model=self.llm,
            tools=[self.agent_tools.get_web_search_tool()],
            name="web_search_agent",
            prompt="""
                당신은 부동산 정보 검색 전문가입니다. 사용자의 질의를 웹 검색에 최적화된 형태로 확장해주세요.

                **확장 시 고려사항:**
                - 최신 뉴스, 시장 동향과 관련된 키워드 포함
                - 구체적인 부동산 정보 및 투자 관련 용어 추가
                - 시기적 정보 (2024년, 2025년, 최근, 최신) 포함
                - 정책 변화, 규제 완화, 개발 허가 등 관련 키워드 추가

                **주요 확장 키워드:**
                - 부동산 시장, 투자 전망, 개발 현황
                - 분양, 입주, 준공, 착공, 승인
                - 규제 완화, 정책 변화, 개발 허가
                - 시세 변동, 거래량, 투자 수익률
            """
        )
    
    def _create_supervisor(self):
        """Supervisor Agent 생성"""
        agent_list = list(self.agents.values())
        
        self.supervisor = create_supervisor(
            model=self.llm,
            agents=agent_list,
            prompt="""
               You are a supervisor managing three agents:
               - country_plan_search_agent: A national comprehensive plan search expert who first checks all real estate development queries to obtain planning information related to domestic real estate investment.
               - city_plan_search_agent: A city/province basic plan search expert with knowledge of regional development basic plans for regional investment at the city/province/county level.
               - web_search_agent: An internet search expert who finds information missing or insufficient in national and local development plans.
               
               To answer real estate investment queries, you first ask the country_plan_search_agent if there are any national development plans.
               If the query contains a location name, you ask the city_plan_search_agent about development plans for that region.
               For development progress, detailed policies, and related news, you ask the web_search_agent.
               
               Based on the compiled information, you provide the optimal answer to the user.
            """
        ).compile()
    
    def get_agent(self, agent_name: str):
        """특정 Agent 반환"""
        return self.agents.get(agent_name)
    
    def get_supervisor(self):
        """Supervisor Agent 반환"""
        return self.supervisor
    
    def list_agents(self):
        """사용 가능한 Agent 목록 반환"""
        return list(self.agents.keys())
    
    def save_supervisor_graph(self, filename: str = "supervisor_graph.png"):
        """Supervisor 그래프를 이미지로 저장"""
        if self.supervisor is None:
            print("Supervisor 없음!!")
            return
            
        try:
            graph_img_bytes = self.supervisor.get_graph(xray=True).draw_mermaid_png()
            img = Image.open(io.BytesIO(graph_img_bytes))
            img.save(filename)
            print(f"이미지 저장 완료: {filename}")
        except Exception as e:
            print(f"그래프 저장 실패: {e}")
    
    def chat(self, message: str) -> Dict[str, Any]:
        """사용자 메시지를 Supervisor를 통해 처리"""
        if self.supervisor is None:
            return {"error": "Supervisor가 생성되지 않음"}
            
        try:
            result = self.supervisor.invoke({
                "messages": [{
                    "role": "user",
                    "content": message
                }]
            })
            
            # 출처 정보 분석 추가
            result["source_analysis"] = self._analyze_agent_usage(result)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_agent_usage(self, result: Dict[str, Any]) -> Dict[str, int]:
        """Agent 사용량 분석"""
        source_counts: Dict[str, int] = {}
        
        if "messages" not in result:
            return source_counts
        
        import re
        
        for msg in result["messages"]:
            print(f"🔍 Message type: {type(msg)}, has content: {hasattr(msg, 'content')}, has tool_call_id: {hasattr(msg, 'tool_call_id')}")
            
            # ToolMessage인 경우 내용에서 metadata 추출
            if hasattr(msg, 'content') and hasattr(msg, 'tool_call_id'):
                content = str(msg.content)
                print(f"🔍 Tool message content preview: {content[:200]}...")
                
                # Document 객체들에서 agent_source 추출 (여러 패턴 시도)
                if "agent_source" in content:
                    # 다양한 패턴 시도
                    patterns = [
                        r"'agent_source':\s*'([^']+)'",
                        r'"agent_source":\s*"([^"]+)"',
                        r"agent_source['\"\s]*:\s*['\"]([^'\"]+)['\"]"
                    ]
                    
                    for pattern in patterns:
                        sources = re.findall(pattern, content)
                        for source in sources:
                            source_counts[source] = source_counts.get(source, 0) + 1
                            print(f"🔍 Found agent_source: {source}")
            
            # 메시지 이름으로도 확인 (fallback)
            if hasattr(msg, 'name') and msg.name:
                print(f"🔍 Message name: {msg.name}")
                if "country_plan_search_agent" in msg.name:
                    source_counts["country_plan"] = source_counts.get("country_plan", 0) + 1
                elif "city_plan_search_agent" in msg.name:
                    source_counts["city_plan"] = source_counts.get("city_plan", 0) + 1
                elif "web_search_agent" in msg.name:
                    source_counts["web_search"] = source_counts.get("web_search", 0) + 1
        
        print(f"🔍 Final source_counts: {source_counts}")
        return source_counts
    
    def get_agent_info(self) -> Dict[str, Union[str, List[str]]]:
        """Agent 정보 반환"""
        return {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "agents": self.list_agents(),
            "supervisor_status": "생성됨" if self.supervisor else "미생성"
        }