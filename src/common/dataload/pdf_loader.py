"""
PDF 데이터 로딩을 담당하는 클래스 모듈
"""
import os
import glob
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
import numpy as np

from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from chromadb import PersistentClient
from ..llm_factory import LLMFactory

# fallback을 위한 pypdf import
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("pypdf를 설치하면 더 안정적인 PDF 처리가 가능합니다: pip install pypdf")


class PlanDataLoader:
    """
    PDF 파일들을 로딩하여 벡터DB에 저장 담당하는 클래스
    """
    
    # 클래스 레벨에서 DB 경로 관리
    DB_PATH = "database/chroma"
    
    def __init__(self, embedding_model: Optional[str] = None, use_docling: bool = True):
        self.default_embedding_model = embedding_model or "bge-m3"  # ollama 모델로 변경
        self.use_docling = use_docling
        
        # 데이터베이스 디렉토리 생성
        self._ensure_db_directory()
        
    @staticmethod
    def get_chroma_db(collection_name: str, embedding_model: Optional[str] = None) -> Optional[Chroma] :
        """컬렉션 이름으로 Chroma 컬렉션 객체 반환
        
        Args:
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델명 (기본값: bge-m3)
            
        Returns:
            Collection: Chroma 컬렉션 객체. 컬렉션이 없으면 None 반환
        """
        try:
            # 임베딩 모델 생성
            embed_model = LLMFactory.create_embedding_model(embedding_model or "bge-m3")
            
            # Chroma 객체 생성 및 반환
            from langchain_community.vectorstores import Chroma
            return Chroma(
                embedding_function=embed_model,
                collection_name=collection_name,
                persist_directory=PlanDataLoader.DB_PATH
            )
        except Exception as e:
            print(f"Chroma 컬렉션 생성 실패: {e}")
            return None
    
    def _ensure_db_directory(self):
        """데이터베이스 디렉토리가 존재하는지 확인하고 생성"""
        os.makedirs(self.DB_PATH, exist_ok=True)
        print(f"📁 Chroma DB 경로: {os.path.abspath(self.DB_PATH)}")
    
    @classmethod
    def get_chroma_client(cls):
        """Chroma DB 클라이언트 반환"""
        cls._ensure_class_db_directory()
        return PersistentClient(path=cls.DB_PATH)
    
    @classmethod
    def _ensure_class_db_directory(cls):
        """클래스 메서드용 DB 디렉토리 확인"""
        os.makedirs(cls.DB_PATH, exist_ok=True)
    
    def _extract_text_with_pypdf(self, pdf_path: str) -> str:
        """pypdf를 사용한 텍스트 추출 (fallback)"""
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf가 설치되지 않음")
        
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_text_with_docling(self, pdf_path: str) -> str:
        """docling을 사용한 텍스트 추출 (메인)"""
        converter = DocumentConverter()
        docling_doc = converter.convert(pdf_path)
        return docling_doc.document.export_to_text()
    
    def load_single_pdf(
        self,
        pdf_path: str,
        chroma_collection: str = "default",
        embedding_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        additional_metadata: Optional[dict] = None,
        **kwargs
    ) -> int:

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없음: {pdf_path}")

        # 1. PDF에서 텍스트 추출 (docling 우선, 실패시 pypdf 사용)
        text = None
        extraction_method = "unknown"
        
        if self.use_docling:
            try:
                print(f"  docling으로 텍스트 추출 시도...")
                text = self._extract_text_with_docling(pdf_path)
                extraction_method = "docling"
                print(f"  docling 추출 성공!")
            except Exception as e:
                print(f"  docling 추출 실패: {e}")
                if PYPDF_AVAILABLE:
                    try:
                        print(f"  pypdf로 fallback 시도...")
                        text = self._extract_text_with_pypdf(pdf_path)
                        extraction_method = "pypdf"
                        print(f"  pypdf 추출 성공!")
                    except Exception as e2:
                        print(f"  pypdf 추출도 실패: {e2}")
                        raise ValueError(f"모든 PDF 추출 방법이 실패함. docling: {e}, pypdf: {e2}")
                else:
                    raise ValueError(f"docling 추출 실패하고 pypdf가 설치되지 않음: {e}")
        else:
            # pypdf만 사용
            if PYPDF_AVAILABLE:
                text = self._extract_text_with_pypdf(pdf_path)
                extraction_method = "pypdf"
            else:
                raise ValueError("pypdf가 설치되지 않았고 docling 사용이 비활성화됨")
        
        if not text or not text.strip():
            raise ValueError("PDF에서 텍스트를 추출하지 못함")
        
        # 기본 메타데이터 설정
        base_metadata = {
            "source": pdf_path,
            "extraction_method": extraction_method
        }
        # 추가 메타데이터가 있으면 병합
        if additional_metadata:
            base_metadata.update(additional_metadata)
        
        docs = [Document(page_content=text, metadata=base_metadata)]

        # 2. 텍스트 분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents(docs)
        if not split_docs:
            raise ValueError("텍스트 분할 결과가 없음")

        # 3. 임베딩 모델 준비
        model_name = embedding_model or self.default_embedding_model
        try:
            embed_model = LLMFactory.create_embedding_model(model_name)
        except Exception as e:
            print(f"임베딩 모델 생성 실패: {e}")
            # ollama bge-m3 모델로 fallback
            print("ollama bge-m3 모델로 fallback 시도...")
            embed_model = LLMFactory.create_embedding_model("bge-m3", provider="ollama")

        # 4. Chroma DB 연결 및 컬렉션 준비
        chroma_client = self.get_chroma_client()
        
        # 임베딩 함수를 포함한 컬렉션 생성
        try:
            collection = chroma_client.get_collection(name=chroma_collection)
            print(f"  기존 컬렉션 '{chroma_collection}' 사용")
        except Exception:
            # 컬렉션이 없으면 새로 생성 (임베딩 함수 없이)
            collection = chroma_client.create_collection(name=chroma_collection)
            print(f"  새 컬렉션 '{chroma_collection}' 생성")

        # 5. 임베딩 및 저장
        texts = [doc.page_content for doc in split_docs]
        try:
            embeddings = np.array(embed_model.embed_documents(texts), dtype=np.float32)
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
            raise
        
        ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(texts))]
        
        # 각 청크에 동일한 메타데이터 할당
        from typing import cast
        metadatas = cast(list, [doc.metadata for doc in split_docs])
        
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return len(texts)
    
    def load_pdf_directory(
        self,
        pdf_dir: str,
        chroma_collection: str,
        is_city_file: bool = False,
        embedding_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        result = {
            "success_count": 0,
            "fail_count": 0,
            "total_chunks": 0,
            "processed_files": [],
            "failed_files": []
        }
        
        if not pdf_dir or not os.path.exists(pdf_dir):
            print(f"PDF 디렉토리가 존재하지 않거나 설정되지 않음: {pdf_dir}")
            return result
        
        # 디렉토리 내의 모든 PDF 파일 찾기
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        print(f"발견된 PDF 파일 수: {len(pdf_files)}")
        
        for pdf_file in pdf_files:
            try:
                print(f"처리 중: {pdf_file}")
                
                # 메타데이터 준비
                additional_metadata = None
                city_name = None
                
                if is_city_file:
                    # PDF 파일명에서 도시명 추출 (파일명을 '_'로 구분했을 때 첫 번째 단어)
                    pdf_filename = os.path.basename(pdf_file)
                    pdf_name_without_ext = os.path.splitext(pdf_filename)[0]  # 확장자 제거
                    city_name = pdf_name_without_ext.split('_')[0]  # '_'로 구분한 첫 번째 단어
                    additional_metadata = {"city_name": city_name}
                
                # PDF 처리 및 저장
                chunk_count = self.load_single_pdf(
                    pdf_path=pdf_file,
                    chroma_collection=chroma_collection,
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    additional_metadata=additional_metadata,
                    **kwargs
                )
                
                # 결과 기록
                result["success_count"] += 1
                result["total_chunks"] += chunk_count
                result["processed_files"].append({
                    "file": pdf_file,
                    "chunks": chunk_count,
                    "city_name": city_name if is_city_file else None
                })
                
                if is_city_file:
                    print(f"  저장된 청크 수: {chunk_count}, 도시명: {city_name}")
                else:
                    print(f"  저장된 청크 수: {chunk_count}")
                    
            except Exception as e:
                print(f"PDF 파일 변환 중 오류 발생 ({pdf_file}): {e}")
                result["fail_count"] += 1
                result["failed_files"].append({"file": pdf_file, "error": str(e)})
                continue
        
        print(f"처리 완료 - 성공: {result['success_count']}, 실패: {result['fail_count']}, 총 청크: {result['total_chunks']}")
        return result

    @staticmethod
    def get_chroma_collection(collection_name: str):
        """지정된 컬렉션 반환"""
        client = PlanDataLoader.get_chroma_client()
        try:
            return client.get_collection(name=collection_name)
        except Exception as e:
            print(f"컬렉션 '{collection_name}'을 찾을 수 없음: {e}")
            return None

    @staticmethod
    def list_collections():
        """모든 컬렉션 목록 반환"""
        client = PlanDataLoader.get_chroma_client()
        collections = client.list_collections()
        print(f"📋 사용 가능한 컬렉션 ({len(collections)}개):")
        for collection in collections:
            count = collection.count()
            print(f"  - {collection.name}: {count}개 문서")
        return collections
    
    @staticmethod
    def reset_chroma_db(chroma_collection: Optional[str] = None):
        """
        Chroma DB 초기화
        """
        client = PlanDataLoader.get_chroma_client()
        if chroma_collection:
            # 지정한 컬렉션만 삭제
            collections = [col.name for col in client.list_collections()]
            if chroma_collection in collections:
                client.delete_collection(name=chroma_collection)
                print(f"컬렉션 '{chroma_collection}' 삭제 완료")
            else:
                print(f"컬렉션 '{chroma_collection}'이(가) 존재하지 않음")
        else:
            # 전체 컬렉션 삭제
            collections = client.list_collections()
            for col in collections:
                client.delete_collection(name=col.name)
            print(f"전체 컬렉션 ({len(collections)}개) 삭제 완료")
    
    @staticmethod
    def get_collection_info(collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        try:
            collection = PlanDataLoader.get_chroma_collection(collection_name)
            if collection is None:
                return {"error": f"컬렉션 '{collection_name}'을 찾을 수 없음"}
            
            count = collection.count()
            # 샘플 데이터 몇 개 가져오기
            sample_results = collection.peek(limit=3)
            
            return {
                "name": collection_name,
                "count": count,
                "sample_ids": sample_results.get('ids', []),
                "sample_metadatas": sample_results.get('metadatas', [])
            }
        except Exception as e:
            return {"error": str(e)} 