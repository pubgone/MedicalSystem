# backend/app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import ConfigDict  # ✅ 新增导入
from typing import Optional, List
import os

class Settings(BaseSettings):
    # 服务配置
    APP_NAME: str = "医疗 RAG 系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS 配置（支持代理访问的关键）
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8080",      # Docker 前端
        "http://127.0.0.1:8080",
        "http://localhost:5500",      # ✅ Live Server 默认端口
        "http://127.0.0.1:5500",      # ✅ 添加这个
        "http://localhost:5501",      # ✅ 如果 5500 被占用会用 5501
        "http://127.0.0.1:5501",
        "http://localhost:8000",      # 后端直连
        "*",                          # 开发环境临时允许所有
    ]
    ALLOW_CREDENTIALS: bool = True
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # 模型配置（敏感信息通过环境变量或 .env 文件配置）
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh"
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    LLM_NAME: str = "qwen-plus"
    LLM_ENDPOINT: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_KEY: Optional[str] = None  # 必须通过环境变量或 .env 设置
    DEVICE: str = "cuda"
    
    # 向量库配置
    CHROMA_PERSIST_DIR: str = "backend/chroma_db"
    COLLECTION_NAME: str = "medical_knowledge_v1"
    
    # RAG 配置
    MAX_CONTEXT_LENGTH: int = 4000
    DEFAULT_TOP_K: int = 5
    DEFAULT_RERANK_TOP_K: int = 3
    
    # 安全配置
    API_KEY: Optional[str] = "1234"  # 可选的 API 密钥
    RATE_LIMIT: int = 60  # 每分钟请求限制
    
    # 代理配置（支持反向代理）
    PROXY_PREFIX: str = ""  # Nginx 代理时使用，如 "/api"
    TRUSTED_PROXIES: List[str] = ["127.0.0.1", "localhost"]
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # ✅ 关键：忽略 .env 中未定义的字段
    )

settings = Settings()