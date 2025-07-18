from typing import List
from langchain_core.documents import Document
from langchain.agents import tool
from langchain_core.messages import ToolMessage
from langchain_tavily import TavilySearch
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langgraph.prebuilt import InjectedState
from common.llm_factory import LLMFactory
from common.dataload.pdf_loader import PlanDataLoader
from langgraph.graph import MessagesState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from typing import Annotated


class AgentTools:
    """Tool들을 관리하는 클래스"""
    
    def __init__(self, embedding_model: str = "bge-m3"):
        """
        AgentTools 초기화
        
        Args:
            embedding_model: 사용할 임베딩 모델명
        """
        self.embedding_model = embedding_model
        self.reranker = self._create_reranker()
        self.country_plan_retriever = self._create_country_plan_retriever()
        self.city_plan_retriever = self._create_city_plan_retriever()
        self.tavily_search_tool = TavilySearch(topic="general",max_results=5)
        
    def _create_reranker(self):
        """재정렬 모델 생성"""
        rerank_model = HuggingFaceCrossEncoder(
            model_name="Alibaba-NLP/gte-multilingual-reranker-base",
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True,
            } 
        )
        return CrossEncoderReranker(model=rerank_model, top_n=4)
    
    def _create_country_plan_retriever(self):
        """국토종합계획 검색기 생성"""
        country_plan_db = PlanDataLoader.get_chroma_db("country_plan", self.embedding_model)
        if country_plan_db is None:
            raise ValueError("❌ 컬렉션 생성 실패: country_plan")
        
        return ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=country_plan_db.as_retriever(search_kwargs={"k": 20})
        )
    
    def _create_city_plan_retriever(self):
        """시도군 기본계획 검색기 생성"""
        city_plan_db = PlanDataLoader.get_chroma_db("city_plan", self.embedding_model)
        if city_plan_db is None:
            raise ValueError("❌ 컬렉션 생성 실패: city_plan")
        
        return ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=city_plan_db.as_retriever(search_kwargs={"k": 20})
        )
    
    def get_country_plan_search_tool(self):
        """국토종합계획 검색 도구 반환"""
        @tool
        def country_plan_search(query: str) -> List[Document]:
            """
            지역 부동산 개발 계획의 상위 정책인 국토종합계획에서 검색
            
            Args:
                query: 검색 쿼리
                
            Returns:
                List[Document]: 검색 결과
            """
            docs = self.country_plan_retriever.invoke(query)
            
            print(f"country_plan_search: {docs}, count: {len(docs)}")
            
            # 모든 문서에 source 정보 추가
            for doc in docs:
                doc.metadata["agent_source"] = "country_plan"
                doc.metadata["agent_name"] = "국토종합계획"
            
            if len(docs) > 0:
                return docs
            
            return [Document(page_content="국토종합계획에서 검색된 결과가 없습니다.", 
                           metadata={"agent_source": "country_plan", "agent_name": "국토종합계획"})]
        
        return country_plan_search
    
    def get_city_plan_search_tool(self):
        """시도군 기본계획 검색 도구 반환"""
        @tool
        def city_plan_search(query: str, city_name: str = "_ANY_") -> List[Document]:
            """
            지역 부동산 개발 계획이 담긴 시도군 기본계획에서 검색
            
            Args:
                query: 검색 쿼리
                city_name: 검색할 도시 이름
                    - 도시 이름이 "영월"인 경우 "youngwol"
                    - 도시 이름이 "인천"인 경우 "incheon"
                    - 도시 이름이 "서울"인 경우 "seoul"
                    - 도시 이름을 특정하지 않거나 모든 도시에 대해 검색하려면 "_ANY_" 사용
                
            Returns:
                List[Document]: 검색 결과
            """
            docs = []
            if city_name == "_ANY_":
                docs = self.city_plan_retriever.invoke(query)
            else:
                # city_name 필터링 위해 search_kwargs 사용
                docs = self.city_plan_retriever.invoke(query, filter={"city_name": city_name})
                
            print(f"city_plan_search: {docs}, count: {len(docs)}")
            
            # 모든 문서에 source 정보 추가
            for doc in docs:
                doc.metadata["agent_source"] = "city_plan"
                doc.metadata["agent_name"] = "시도군 기본계획"
            
            if len(docs) > 0:
                return docs
            
            return [Document(page_content="시도군 기본계획에서 검색된 결과가 없습니다.", 
                           metadata={"agent_source": "city_plan", "agent_name": "시도군 기본계획", "city_name": city_name})]
        
        return city_plan_search
    
    def get_web_search_tool(self):
        """웹 검색 도구 반환"""
        @tool
        def tavily_search(query: str) -> List[Document]:
            """
            인터넷에서 검색
            
            Args:
                query: 검색 쿼리
                
            Returns:
                List[Document]: 검색 결과
            """
            docs = self.tavily_search_tool.invoke(query)
            print(f"tavily_search: {docs}, count: {len(docs)}")
            result_docs = []
            for doc in docs:
                if isinstance(doc, Document):
                    doc.metadata["agent_source"] = "web_search"
                    doc.metadata["agent_name"] = "웹 검색"
                    result_docs.append(doc)
                elif isinstance(doc, str):
                    # 문자열이면 Document로 변환
                    result_docs.append(Document(page_content=doc, metadata={"agent_source": "web_search", "agent_name": "웹 검색"}))
                else:
                    # 예외 상황: 기타 타입
                    result_docs.append(Document(page_content=str(doc), metadata={"agent_source": "web_search", "agent_name": "웹 검색"}))
            return result_docs
        
        return tavily_search
            
    
    def get_all_tools(self):
        """모든 도구 반환"""
        return [
            self.get_country_plan_search_tool(),
            self.get_city_plan_search_tool(),
            self.get_web_search_tool()
        ]