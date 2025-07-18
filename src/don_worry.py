import operator
import os
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from common.llm_factory import LLMFactory
from common.dataload.pdf_loader import PlanDataLoader
import gradio as gr

from dotenv import load_dotenv

# .env 파일 로드함 (파일이 없어도 에러 발생하지 않음)
load_dotenv()


def setup_environment():
    """환경변수 설정 확인 및 기본값 설정"""
    # 기본 PDF 디렉토리 설정
    country_pdf_dir = os.getenv("COUNTRY_PLAN_PDF_DIR", "raw_files/country_docs")
    city_pdf_dir = os.getenv("CITY_PLAN_PDF_DIR", "raw_files/city_docs")
    reset_db = os.getenv("RESET_DB", "false").lower() == "true"
    
    # 처리 모드 설정 (fast: pypdf 우선, normal: docling 우선)
    process_mode = os.getenv("PDF_PROCESS_MODE", "fast").lower()
    
    # 임베딩 모델 설정
    embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3")
    
    print("🔧 환경 설정:")
    print(f"  - 국토계획 PDF 디렉토리: {country_pdf_dir}")
    print(f"  - 시도군 PDF 디렉토리: {city_pdf_dir}")
    print(f"  - DB 리셋: {reset_db}")
    print(f"  - 처리 모드: {process_mode} ({'pypdf 우선' if process_mode == 'fast' else 'docling 우선'})")
    print(f"  - 임베딩 모델: {embedding_model}")
    
    return country_pdf_dir, city_pdf_dir, reset_db, process_mode, embedding_model


def check_ollama_connection():
    """Ollama 연결 확인"""
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"🤖 Ollama 연결 확인 중... ({ollama_url})")
        # 실제 연결 테스트는 임베딩 모델 생성할 때 확인됨
        return True
    except Exception as e:
        print(f"⚠️  Ollama 연결 실패: {e}")
        print("   Ollama가 실행 중인지 확인하세요: ollama serve")
        return False


def reload_db(need_reset: bool = False, country_pdf_dir: str = "", city_pdf_dir: str = "", embedding_model: str = "bge-m3", process_mode: str = "fast"):
    """데이터베이스 리로드"""
    if not need_reset:
        print("🔄 DB 리셋이 설정되지 않음. 기존 데이터 유지.")
        return
    
    print("..DB 재구축 중..")
    PlanDataLoader.reset_chroma_db()
    
    # PDF 데이터 로더 초기화
    use_docling = (process_mode == "normal")
    pdf_loader = PlanDataLoader(
        embedding_model=embedding_model,
        use_docling=use_docling
    )
    
    # 국토종합계획 파일 처리
    if country_pdf_dir and os.path.exists(country_pdf_dir):
        print(f"\n📄 국토종합계획 PDF 처리 시작")
        print(f"   디렉토리: {country_pdf_dir}")
        
        try:
            country_result = pdf_loader.load_pdf_directory(
                pdf_dir=country_pdf_dir,
                chroma_collection="country_plan",
                embedding_model=embedding_model,
                is_city_file=False,
                chunk_size=1000,
                chunk_overlap=20
            )
            print(f"✅ 국토종합계획 처리 결과:")
            print(f"   성공: {country_result['success_count']}, 실패: {country_result['fail_count']}")
            print(f"   총 청크: {country_result['total_chunks']}")
            if country_result['failed_files']:
                print("   실패한 파일들:")
                for failed in country_result['failed_files']:
                    print(f"     - {failed['file']}: {failed['error']}")
        except Exception as e:
            print(f"❌ 국토종합계획 처리 중 전체 오류: {e}")
    else:
        print(f"⚠️  국토종합계획 디렉토리가 존재하지 않음: {country_pdf_dir}")
    
    # 시도군 기본계획 파일 처리
    if city_pdf_dir and os.path.exists(city_pdf_dir):
        print(f"\n🏙️  시도군 기본계획 PDF 처리 시작")
        print(f"   디렉토리: {city_pdf_dir}")
        
        try:
            city_result = pdf_loader.load_pdf_directory(
                pdf_dir=city_pdf_dir,
                chroma_collection="city_plan",
                embedding_model=embedding_model,
                is_city_file=True,
                chunk_size=1000,
                chunk_overlap=20
            )
            print(f"✅ 시도군 기본계획 처리 결과:")
            print(f"   성공: {city_result['success_count']}, 실패: {city_result['fail_count']}")
            print(f"   총 청크: {city_result['total_chunks']}")
            if city_result['failed_files']:
                print("   실패한 파일들:")
                for failed in city_result['failed_files']:
                    print(f"     - {failed['file']}: {failed['error']}")
        except Exception as e:
            print(f"❌ 시도군 기본계획 처리 중 전체 오류: {e}")
    else:
        print(f"⚠️  시도군 기본계획 디렉토리가 존재하지 않음: {city_pdf_dir}")


def test_query(query: str, collection_name: str = "country_plan", n_results: int = 5, embedding_model: str = "bge-m3", where: Optional[dict] = None):
    """컬렉션에서 쿼리 테스트"""
    print(f"\n🔍 쿼리 테스트: '{query}'")
    print(f"   컬렉션: {collection_name}")
    print(f"   임베딩 모델: {embedding_model}")
    print(f"   필터: {where}")
    
    try:
        collection = PlanDataLoader.get_chroma_collection(collection_name)
        if collection is None:
            print(f"❌ 컬렉션 '{collection_name}'을 찾을 수 없습니다.")
            return
        
        # 쿼리용 임베딩 모델 생성
        try:
            embed_model = LLMFactory.create_embedding_model(embedding_model)
        except Exception as e:
            print(f"임베딩 모델 생성 실패: {e}")
            # ollama bge-m3 모델로 fallback
            print("ollama bge-m3 모델로 fallback 시도...")
            embed_model = LLMFactory.create_embedding_model("bge-m3", provider="ollama")
        
        # 쿼리 임베딩 생성
        query_embedding = embed_model.embed_documents([query])[0]
        
        # 컬렉션에서 검색 (where 필터 적용)
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        
        #### 음... 지정하면 난리 나네..
        if where:
            query_params["where"] = where
            
        results = collection.query(**query_params)
        
        # 결과 안전성 확인
        ids = results.get('ids', [[]])
        distances = results.get('distances', [[]])
        documents = results.get('documents', [[]])
        metadatas = results.get('metadatas', [[]])
        
        if ids and len(ids) > 0 and ids[0]:
            result_count = len(ids[0])
            print(f"\n📊 검색 결과 ({result_count}개):")
            
            for i, (doc_id, distance, document, metadata) in enumerate(zip(
                ids[0],
                distances[0] if distances and len(distances) > 0 else [0] * result_count,
                documents[0] if documents and len(documents) > 0 else [''] * result_count,
                metadatas[0] if metadatas and len(metadatas) > 0 else [{}] * result_count
            )):
                print(f"\n{i+1}. 문서 ID: {doc_id}")
                print(f"   유사도: {1-distance:.3f}")
                print(f"   메타데이터: {metadata}")
                print(f"   내용 (처음 200자): {document[:200]}...")
                
                if where:
                    print(f"metadata.get('city_name'): {metadata.get('city_name')} , where.get('city_name'): {where.get('city_name')}")
                    print(f"환장파티... 필터 일치: {'✅' if metadata.get('city_name') == where.get('city_name') else '❌'} (검색:{where.get('city_name')}, 문서:{metadata.get('city_name')})")
                    
            
        else:
            print("   검색 결과가 없습니다.")
            
    except Exception as e:
        print(f"❌ 쿼리 실행 중 오류: {e}")


def show_db_status():
    """데이터베이스 상태 표시"""
    print("\n=== 데이터베이스 상태: ===")
    try:
        collections = PlanDataLoader.list_collections()
        if not collections:
            print("   생성된 컬렉션이 없습니다.")
        else:
            for collection in collections:
                info = PlanDataLoader.get_collection_info(collection.name)
                if 'error' not in info:
                    print(f"\n📋 컬렉션: {info['name']}")
                    print(f"   문서 수: {info['count']}개")
                    if info['sample_metadatas']:
                        print(f"   샘플 메타데이터: {info['sample_metadatas'][0]}")
    except Exception as e:
        print(f"   상태 확인 중 오류: {e}")


# main 함수 정의
def main():
    # 환경 설정
    country_pdf_dir, city_pdf_dir, need_reset_db, process_mode, embedding_model = setup_environment()
    
    # Ollama 연결 확인 (bge-m3 모델 사용 시에만)
    if embedding_model == "bge-m3":
        if not check_ollama_connection():
            return
    
    # 데이터베이스 상태 확인
    show_db_status()
    
    # 데이터베이스 리로드 (필요시)
    reload_db(need_reset_db, country_pdf_dir, city_pdf_dir, embedding_model, process_mode)    
    
    # 테스트 쿼리 실행
    print("\n" + "="*50)
    # 일반 텍스트 검색
    # test_query("부산 광역시 교통망 계획", "country_plan", 5, embedding_model)
    # test_query("도시 계획", "city_plan", 5, embedding_model)
    
    # # 도시명으로 검색 
    # test_query("도로 계획", "city_plan", 5, embedding_model)
    # test_query("도로 계획", "city_plan", 5, embedding_model, where={"city_name": "youngwol"})
    # test_query("도로 계획", "city_plan", 5, embedding_model, where={"city_name": "영월"}) #.. 안돼... 망할...
    # test_query("도로 계획", "city_plan", 5, embedding_model, where={"city_name": {"$eq":"영월"}}) #.. 안돼... 망할...
    
    
    ###------- Agent 시스템 초기화
    from common.agent.agent_manager import AgentManager
    
    print("\n🤖 Agent 시스템 초기화 중...")
    try:
        # AgentManager 생성 - 모든 Agent와 Tool들을 자동으로 설정
        agent_manager = AgentManager(
            embedding_model=embedding_model,
            llm_model="gpt-4.1-mini"
        )
        
        # Agent 정보 출력
        agent_info = agent_manager.get_agent_info()
        print(f"   임베딩 모델: {agent_info['embedding_model']}")
        print(f"   LLM 모델: {agent_info['llm_model']}")
        print(f"   생성된 Agent들: {', '.join(agent_info['agents'])}")
        print(f"   Supervisor 상태: {agent_info['supervisor_status']}")
        
        # Supervisor 그래프 저장
        agent_manager.save_supervisor_graph()
        
        print("✅ Agent 시스템 초기화 완료!")
        
        # Gradio 채팅 앱 생성 및 실행
        create_gradio_app(agent_manager)
        
    except Exception as e:
        print(f"❌ Agent 시스템 초기화 실패: {e}")
        return

def analyze_messages_for_sources(messages):
    """메시지에서 직접 agent 사용량 분석 (fallback)"""
    agent_counts = {"country_plan": 0, "city_plan": 0, "web_search": 0}
    
    for msg in messages:
        msg_str = str(msg)
        
        # 메시지 이름으로 확인
        if hasattr(msg, 'name') and msg.name:
            if "country_plan_search_agent" in msg.name:
                agent_counts["country_plan"] += 1
            elif "city_plan_search_agent" in msg.name: 
                agent_counts["city_plan"] += 1
            elif "web_search_agent" in msg.name:
                agent_counts["web_search"] += 1
        
        # 메시지 내용으로도 확인 (더 광범위)
        if "country_plan_search" in msg_str.lower() or "국토종합계획" in msg_str:
            agent_counts["country_plan"] += 1
        if "city_plan_search" in msg_str.lower() or ("시도군" in msg_str and "계획" in msg_str):
            agent_counts["city_plan"] += 1  
        if "tavily_search" in msg_str.lower() or "web_search" in msg_str.lower():
            agent_counts["web_search"] += 1
    
    # Agent별 아이콘과 이름 매핑
    agent_info = {
        "country_plan": {"icon": "🏛️", "name": "국토종합계획 분석"},
        "city_plan": {"icon": "🏙️", "name": "시도군 기본계획 분석"},
        "web_search": {"icon": "🌐", "name": "웹 검색 정보"}
    }
    
    sources = []
    for agent_source, count in agent_counts.items():
        if count > 0:
            info = agent_info[agent_source]
            # 중복 카운트 방지 - 최대 1로 제한
            final_count = min(count, 3)  # 너무 많지 않게 제한
            sources.append(f"{info['icon']} {info['name']} ({final_count}건)")
    
    return " | ".join(sources) if sources else ""


def extract_answer_with_sources(result):
    """결과에서 사용된 agent들의 출처 정보 추출 (metadata 기반)"""
    if "source_analysis" not in result:
        return ""
    
    source_analysis = result["source_analysis"]
    
    # Agent별 아이콘과 이름 매핑
    agent_info = {
        "country_plan": {"icon": "🏛️", "name": "국토종합계획 분석"},
        "city_plan": {"icon": "🏙️", "name": "시도군 기본계획 분석"},
        "web_search": {"icon": "🌐", "name": "웹 검색 정보"}
    }
    
    # 사용된 agent들만 출처로 표시
    sources = []
    for agent_source, count in source_analysis.items():
        if count > 0 and agent_source in agent_info:
            info = agent_info[agent_source]
            sources.append(f"{info['icon']} {info['name']} ({count}건)")
    
    return " | ".join(sources) if sources else ""


def create_gradio_app(agent_manager):
    """Gradio 채팅 앱 생성 및 실행"""
    
    def chat_with_supervisor(message, history):
        """supervisor agent와 채팅하는 함수"""
        try:
            # agent_manager를 통해 supervisor와 대화
            result = agent_manager.chat(message)
            
            if "error" in result:
                return f"❌ 오류가 발생했습니다: {result['error']}"
            
            # messages에서 마지막 메시지 추출
            answer = ""
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    answer = last_message.content
                else:
                    answer = str(last_message)
            
            if not answer:
                return "응답을 받지 못했습니다."
            
            # 출처 정보 추가 (간단한 메시지 분석 우선 사용)
            sources = analyze_messages_for_sources(result.get("messages", []))
            print(f"🔍 Direct message analysis sources: '{sources}'")
            
            # metadata 기반 분석도 시도
            if not sources:
                sources = extract_answer_with_sources(result)
                print(f"🔍 Metadata-based sources: '{sources}'")
            
            if sources:
                answer += "\n\n---\n**📋 참고 정보:**\n" + sources
            
            return answer
            
        except Exception as e:
            return f"❌ 채팅 중 오류 발생: {str(e)}"
    
    # Gradio ChatInterface 생성
    chat_interface = gr.ChatInterface(
        chat_with_supervisor,
        title="🏘️ Don-Worry 부동산 투자 상담 챗봇",
        description="""
        **국토종합계획 & 시도군 기본계획 기반 부동산 투자 상담 서비스**
        
        🔍 **주요 기능:**
        - 국토종합계획 검색 및 분석
        - 시도군 기본계획 정보 제공 (현재는 영월시만 있어용)
        - 웹 검색을 통한 최신 정보 수집
        - 종합적인 부동산 투자 가이드
        
        💡 **질문 예시:**
        - "영월시 아파트 투자 전망은 어떤가요?"
        - "부산 강서구 개발 계획이 있나요?"
        - "2025년 이후 투자하기 좋은 지역은 어디인가요?"
        """,
        examples=[
            "영월시에 아파트 투자 계획 중인데 전망이 어떤가요?",
            "부산 지역 개발 계획과 투자 기회를 알려주세요",
            "2020년 내 투자 유망 지역을 추천해주세요",
            "강원도 지역 부동산 시장 전망은 어떤가요?",
            "재개발 예정 지역 정보를 알려주세요"
        ],
        chatbot=gr.Chatbot(
            height=600,
            show_label=False
        ),
        textbox=gr.Textbox(
            placeholder="부동산 투자 관련 질문을 입력하세요...",
            container=False,
            scale=7
        )
    )
    
    # Gradio 실행
    chat_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 공개 링크 생성 안함
        show_error=True,
        quiet=False
    )
    
    


if __name__ == "__main__":
    main()
    
    