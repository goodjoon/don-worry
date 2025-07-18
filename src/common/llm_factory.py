"""
LLM Factory: 다양한 LLM 및 임베딩 모델을 위한 공통 팩토리 클래스

지원하는 모델:
- OpenAI: GPT-4.1-mini, GPT-4.1, GPT-4o-mini, text-embedding-3-small
- Ollama: Qwen3:0.6b, bge-m3, nomic-embed-text
- Google Gemini: Gemini-1.5-flash, Gemini-2.0-flash
- HuggingFace: BAAI/bge-m3
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

class LLMFactory:
    """
    다양한 LLM 및 임베딩 모델 제공업체의 모델을 생성하는 팩토리 클래스
    """
    
    @staticmethod
    def create_llm(
        model_name: str,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        LLM 클라이언트를 생성함
        """
        if provider is None:
            provider = LLMFactory._detect_provider(model_name)
        provider = provider.lower()
        if provider == "openai":
            return LLMFactory._create_openai_llm(model_name, temperature, max_tokens, **kwargs)
        elif provider == "ollama":
            return LLMFactory._create_ollama_llm(model_name, temperature, max_tokens, **kwargs)
        elif provider == "gemini" or provider == "google":
            return LLMFactory._create_gemini_llm(model_name, temperature, max_tokens, **kwargs)
        elif provider == "huggingface":
            return LLMFactory._create_huggingface_llm(model_name, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 제공업체임: {provider}")

    @staticmethod
    def create_embedding_model(model_name: str, provider: Optional[str] = None, **kwargs):
        """
        임베딩 모델 클라이언트 생성 함수
        """
        if provider is None:
            provider = LLMFactory._detect_embedding_provider(model_name)
        provider = provider.lower()
        if provider == "openai":
            return LLMFactory._create_openai_embedding(model_name, **kwargs)
        elif provider == "ollama":
            return LLMFactory._create_ollama_embedding(model_name, **kwargs)
        elif provider == "huggingface":
            return LLMFactory._create_huggingface_embedding(model_name, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 임베딩 제공업체임: {provider}")

    @staticmethod
    def _detect_provider(model_name: str) -> str:
        model_name = model_name.lower()
        openai_models = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "text-davinci"]
        if any(model in model_name for model in openai_models):
            return "openai"
        gemini_models = ["gemini", "bard", "gemini-1.5-flash", "gemini-2.0-flash"]
        if any(model in model_name for model in gemini_models):
            return "gemini"
        huggingface_models = ["baai/bge-m3"]
        if any(model in model_name for model in huggingface_models):
            return "huggingface"
        ollama_models = ["llama", "qwen3", "bge-m3", "nomic-embed-text"]
        if any(model in model_name for model in ollama_models):
            return "ollama"
        return "ollama"

    @staticmethod
    def _detect_embedding_provider(model_name: str) -> str:
        model_name = model_name.lower()
        if model_name in ["text-embedding-3-small"]:
            return "openai"
        if model_name in ["baai/bge-m3"]:
            return "huggingface"
        if model_name in ["bge-m3", "nomic-embed-text", "qwen3:0.6b"]:
            return "ollama"
        return "openai"

    @staticmethod
    def _create_openai_llm(model_name: str, temperature: float, max_tokens: Optional[int], **kwargs):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai 패키지가 필요함: pip install langchain-openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않음.")
        params = {
            "model": model_name,
            "temperature": temperature,
            "api_key": api_key,
            **kwargs
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        return ChatOpenAI(**params)

    @staticmethod
    def _create_ollama_llm(model_name: str, temperature: float, max_tokens: Optional[int], **kwargs):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("langchain-ollama 패키지가 필요함: pip install langchain-ollama")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        params = {
            "model": model_name,
            "temperature": temperature,
            "base_url": base_url,
            **kwargs
        }
        if max_tokens:
            params["num_predict"] = max_tokens
        return ChatOllama(**params)

    @staticmethod
    def _create_gemini_llm(model_name: str, temperature: float, max_tokens: Optional[int], **kwargs):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("langchain-google-genai 패키지가 필요함: pip install langchain-google-genai")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않음.")
        params = {
            "model": model_name,
            "temperature": temperature,
            "google_api_key": api_key,
            **kwargs
        }
        if max_tokens:
            params["max_output_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**params)

    @staticmethod
    def _create_huggingface_llm(model_name: str, temperature: float, max_tokens: Optional[int], **kwargs):
        try:
            from langchain_huggingface import ChatHuggingFace
        except ImportError:
            raise ImportError("langchain-huggingface 패키지가 필요함: pip install langchain-huggingface")
        params = {
            "model": model_name,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            params["max_new_tokens"] = max_tokens
        return ChatHuggingFace(**params)

    @staticmethod
    def _create_openai_embedding(model_name: str, **kwargs):
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError("langchain-openai 패키지가 필요함: pip install langchain-openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않음.")
        params = {
            "model": model_name,
            "api_key": api_key,
            **kwargs
        }
        return OpenAIEmbeddings(**params)

    @staticmethod
    def _create_ollama_embedding(model_name: str, **kwargs):
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError("langchain-ollama 패키지가 필요함: pip install langchain-ollama")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        params = {
            "model": model_name,
            "base_url": base_url,
            **kwargs
        }
        return OllamaEmbeddings(**params)

    @staticmethod
    def _create_huggingface_embedding(model_name: str, **kwargs):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("langchain-huggingface 패키지가 필요함: pip install langchain-huggingface")
        params = {
            "model": model_name,
            **kwargs
        }
        return HuggingFaceEmbeddings(**params)

    @staticmethod
    def get_available_models() -> Dict[str, list]:
        """
        사용 가능한 LLM/임베딩 모델 목록 반환
        """
        return {
            "openai": [
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o-mini",
                "text-embedding-3-small"
            ],
            "google": [
                "gemini-1.5-flash",
                "gemini-2.0-flash"
            ],
            "ollama": [
                "qwen3:0.6b",
                "bge-m3",
                "nomic-embed-text"
            ],
            "huggingface": [
                "BAAI/bge-m3"
            ]
        }

    @staticmethod
    def list_models():
        models = LLMFactory.get_available_models()
        print("🤖 사용 가능한 LLM/임베딩 모델:")
        print("=" * 50)
        for provider, model_list in models.items():
            print(f"\n📋 {provider.upper()}:")
            for model in model_list:
                print(f"  - {model}")
        print("\n💡 사용 예시:")
        print('  llm = LLMFactory.create_llm("gpt-4.1-mini")')
        print('  embed = LLMFactory.create_embedding_model("bge-m3", provider="ollama")')

# 편의를 위한 함수들
def create_llm(model_name: str, **kwargs):
    return LLMFactory.create_llm(model_name, **kwargs)

def create_embedding_model(model_name: str, **kwargs):
    return LLMFactory.create_embedding_model(model_name, **kwargs)

def list_available_models():
    LLMFactory.list_models()

if __name__ == "__main__":
    list_available_models()
