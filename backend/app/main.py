# backend/app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from contextlib import asynccontextmanager

from .core.config import settings
from .api.routes import router
from .utils.logger import get_logger, setup_logger

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'  # 替换为你的代理端口
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info(f"🚀 启动 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"📍 监听地址：http://{settings.HOST}:{settings.PORT}")
    yield
    # 关闭时
    logger.info("👋 应用关闭")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="医疗知识 RAG 系统 API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ==================== CORS 配置（支持代理访问的关键）====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
    expose_headers=["X-Request-ID", "X-Response-Time"],  # 暴露自定义头
)

# ==================== 中间件 ====================
@app.middleware("http")
async def add_headers(request: Request, call_next):
    """添加请求追踪头"""
    import time
    start_time = time.time()
    
    response = await call_next(request)
    
    # 添加响应时间头
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
    
    return response

# ==================== 路由注册 ====================
app.include_router(router, prefix=settings.PROXY_PREFIX, tags=["API"])

# ==================== 静态文件（前端）====================
# 如果使用独立前端服务，注释掉这部分
# app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== 根路径 ====================
@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

# ==================== 健康检查（兼容 Kubernetes）====================
@app.get("/ready")
async def readiness_check():
    return {"status": "ready"}

@app.get("/live")
async def liveness_check():
    return {"status": "live"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )