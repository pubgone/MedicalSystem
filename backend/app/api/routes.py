# backend/app/api/routes.py
from fastapi import APIRouter, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
from pathlib import Path
from .schemas import (
    ChatRequest, ChatResponse, CitationInfo, QualityAssessment,
    HealthResponse, CollectionStats, APIResponse,
    DocumentUploadRequest, DocumentDeleteRequest,
    StreamChunk, SafetyCheckResult, RetrievalStats,
    EvaluateBatchRequest, EvaluationResponse, EvaluationReport
)
from ..core.config import settings
from ..services.rag_service import RAGService
from ..utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ==================== ✅ 全局变量定义（修复报错）====================
rag_service_instance = None
eval_data_dir = Path(r"backend\evaluation_data")  # ✅ 评估数据目录
feedback_data_dir = Path(r"feedback_data")  # ✅ 反馈数据目录

# 确保目录存在
eval_data_dir.mkdir(parents=True, exist_ok=True)
(eval_data_dir / "results").mkdir(parents=True, exist_ok=True)
feedback_data_dir.mkdir(parents=True, exist_ok=True)
# ==================== ✅ 全局单例（关键修复）====================
rag_service_instance = None

def get_rag_service() -> RAGService:
    """获取 RAG 服务单例"""
    global rag_service_instance
    if rag_service_instance is None:
        rag_service_instance = RAGService()
    return rag_service_instance

# ==================== 健康检查 ====================
@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查接口"""
    service = get_rag_service()
    stats = service.get_collection_stats()
    
    components = {
        "api": True,
        "vector_store": False,
        "retriever": False,
        "llm": False
    }
    errors = []
    
    try:
        # 尝试初始化服务
        from ..services.rag_service import RAGService
        service = RAGService()
        stats = service.get_collection_stats()
        components["vector_store"] = True
        components["retriever"] = True
    except Exception as e:
        error_msg = f"向量库/检索器初始化失败：{str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
    
    try:
        if service:
            components["llm"] = service.check_llm_health()
        else:
            components["llm"] = False
    except Exception as e:
        error_msg = f"LLM 检查失败：{str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
    
    # 如果有错误，返回详细信息
    if errors:
        return HealthResponse(
            status="unhealthy",
            version=settings.APP_VERSION,
            timestamp=datetime.now(),
            components=components
        )
    
    return HealthResponse(
        status="healthy" if all(components.values()) else "degraded",
        version=settings.APP_VERSION,
        timestamp=datetime.now(),
        components=components
    )

# ==================== 聊天接口 ====================
@router.post("/chat", response_model=ChatResponse, tags=["聊天"])
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    x_api_key: Optional[str] = Header(None)
):
    """
    医疗知识问答接口
    """
    # API Key 验证（可选）
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    
    filter_param = request.filter
    if filter_param is not None and len(filter_param) == 0:
        filter_param = None    
    try:
        result = rag_service.chat(
            query=request.query,
            chat_history=request.chat_history,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            filter=request.filter
        )
        
        return ChatResponse(
            success=True,
            query=request.query,
            answer=result["answer"],
            citations=[
                CitationInfo(
                    index=c["index"],
                    source=c["source"],
                    page=c.get("page"),
                    content=c["content"],
                    score=c.get("score")
                )
                for c in result.get("citations", [])
            ],
            quality=QualityAssessment(
                has_citation=result.get("quality", {}).get("has_citation", False),
                has_disclaimer=result.get("quality", {}).get("has_disclaimer", False),
                confidence=result.get("quality", {}).get("confidence", "medium"),
                warnings=result.get("quality", {}).get("warnings", [])
            ),
            stats=result.get("stats", {}),
            timestamp=datetime.now(),
            error=None
        )
    except Exception as e:
        logger.error(f"聊天接口错误：{e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 流式聊天接口 ====================
@router.post("/chat/stream", tags=["聊天"])
async def chat_stream(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    x_api_key: Optional[str] = Header(None)
):
    """
    流式输出聊天接口（SSE）
    """
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    
    filter_param = request.filter
    if filter_param is not None and len(filter_param) == 0:
        filter_param = None    
        
    async def generate():
        try:
            async for chunk in rag_service.chat_stream(
                query=request.query,
                chat_history=request.chat_history,
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                filter=request.filter
            ):
                # SSE 格式
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Nginx 禁用缓冲
        }
    )

# ==================== 向量库管理 ====================
@router.get("/collection/stats", response_model=CollectionStats, tags=["向量库"])
async def get_collection_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """获取向量库统计信息"""
    try:
        stats = rag_service.get_collection_stats()
        return CollectionStats(
            collection_name=stats.get("collection_name", ""),
            total_documents=stats.get("total_documents", 0),
            last_updated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collection/upload", response_model=APIResponse, tags=["向量库"])
async def upload_documents(
    request: DocumentUploadRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """上传文档到向量库"""
    try:
        count = rag_service.upload_documents(
            documents=request.documents,
            collection_name=request.collection_name
        )
        return APIResponse(
            success=True,
            message=f"成功上传 {count} 条文档",
            data={"count": count}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collection/delete", response_model=APIResponse, tags=["向量库"])
async def delete_documents(
    request: DocumentDeleteRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """删除向量库文档"""
    try:
        count = rag_service.delete_documents(
            ids=request.ids,
            filter=request.filter
        )
        return APIResponse(
            success=True,
            message=f"成功删除 {count} 条文档",
            data={"count": count}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 系统信息 ====================
@router.get("/info", response_model=Dict[str, Any], tags=["系统"])
async def get_system_info():
    """获取系统配置信息"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "embedding_model": settings.EMBEDDING_MODEL,
        "rerank_model": settings.RERANK_MODEL,
        "default_top_k": settings.DEFAULT_TOP_K
    }

# ==================== 📊 评估相关接口 ====================

@router.get("/evaluation/questions", tags=["测评"])
async def get_test_questions():
    """获取测试问题集"""
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    questions_file = eval_data_dir / "test_questions.json"
    
    if questions_file.exists():
        with open(questions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

@router.post("/evaluation/evaluate_single", response_model=EvaluationResponse, tags=["测评"])
async def evaluate_single(
    question_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """评估单个问题"""
    try:
        # 加载测试数据
        questions_file = eval_data_dir / "test_questions.json"
        ground_truth_file = eval_data_dir / "ground_truth.json"
        
        if not questions_file.exists():
            return EvaluationResponse(
                success=False,
                message="测试问题集不存在",
                error="NOT_FOUND"
            )
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        question = next((q for q in questions if q["id"] == question_id), None)
        if not question:
            return EvaluationResponse(
                success=False,
                message=f"问题 {question_id} 不存在",
                error="NOT_FOUND"
            )
        
        # 执行 RAG 查询
        result = rag_service.chat(
            query=question["query"],
            retrieval_mode="hybrid_rerank",
            top_k=5
        )
        
        # 简单评分（生产环境应使用 RAGAS）
        score = 0.8 if result.get("quality", {}).get("has_citation", False) else 0.5
        
        return EvaluationResponse(
            success=True,
            message="评估完成",
            data={
                "question_id": question_id,
                "query": question["query"],
                "overall_score": score,
                "answer": result["answer"]
            }
        )
    except Exception as e:
        logger.error(f"单问题评估失败：{e}")
        return EvaluationResponse(
            success=False,
            message="评估失败",
            error=str(e)
        )

@router.post("/evaluation/evaluate_batch", response_model=EvaluationResponse, tags=["测评"])
async def evaluate_batch(
    request: EvaluateBatchRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service)
):
    """批量评估（异步）"""
    try:
        eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        eval_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 后台执行
        background_tasks.add_task(
            _run_batch_evaluation,
            rag_service=rag_service,
            dataset_name=request.dataset_name,
            eval_id=eval_id
        )
        
        return EvaluationResponse(
            success=True,
            message="评估任务已启动",
            data={"evaluation_id": eval_id, "status": "running"}
        )
    except Exception as e:
        logger.error(f"批量评估失败：{e}")
        return EvaluationResponse(
            success=False,
            message="任务启动失败",
            error=str(e)
        )

def _run_batch_evaluation(
    rag_service: RAGService,
    dataset_name: str,
    eval_id: str
):
    """后台执行批量评估"""
    try:
        questions_file = eval_data_dir / "test_questions.json"
        if not questions_file.exists():
            return
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        results = []
        for q in questions:
            result = rag_service.chat(query=q["query"], retrieval_mode="hybrid_rerank", top_k=5)
            results.append({
                "question_id": q.get("id"),
                "query": q["query"],
                "answer": result["answer"],
                "score": 0.8 if result.get("quality", {}).get("has_citation", False) else 0.5
            })
        
        # 保存报告
        report = {
            "evaluation_id": eval_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "avg_score": sum(r["score"] for r in results) / len(results) if results else 0
        }
        
        report_file = eval_data_dir / "results" / f"{eval_id}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估完成：{eval_id}")
    except Exception as e:
        logger.error(f"批量评估执行失败：{e}")

@router.get("/evaluation/reports", tags=["测评"])
async def get_evaluation_reports():
    """获取历史评估报告列表"""
    reports_dir = eval_data_dir / "results"
    if not reports_dir.exists():
        return []
    
    reports = []
    for f in sorted(reports_dir.glob("*.json"), reverse=True):
        with open(f, 'r', encoding='utf-8') as file:
            report = json.load(file)
            reports.append({
                "evaluation_id": report.get("evaluation_id"),
                "timestamp": report.get("timestamp"),
                "status": report.get("status"),
                "avg_score": report.get("avg_score", 0)
            })
    
    return reports

@router.get("/evaluation/report/{evaluation_id}", tags=["测评"])
async def get_evaluation_report(evaluation_id: str):
    """获取指定评估报告详情"""
    report_file = eval_data_dir / "results" / f"{evaluation_id}.json"
    
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    
    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)

@router.post("/evaluation/questions", response_model=APIResponse, tags=["测评"])
async def add_test_question(question: Dict[str, Any]):
    """添加测试问题"""
    try:
        eval_data_dir.mkdir(parents=True, exist_ok=True)
        questions_file = eval_data_dir / "test_questions.json"
        
        questions = []
        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        
        questions.append(question)
        
        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        
        return APIResponse(
            success=True,
            message="问题已添加"
        )
    except Exception as e:
        logger.error(f"添加问题失败：{e}")
        return APIResponse(
            success=False,
            message="添加失败",
            error=str(e)
        )

# ==================== 👥 用户反馈接口 ====================

@router.post("/feedback/submit", response_model=APIResponse, tags=["反馈"])
async def submit_feedback(request: Dict[str, Any]):
    """提交用户反馈"""
    try:
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 保存反馈
        feedback_dir = Path("./feedback_data")
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        feedback = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            **request
        }
        
        feedback_file = feedback_dir / f"{feedback_id}.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, ensure_ascii=False, indent=2)
        
        return APIResponse(
            success=True,
            message="反馈已提交",
            data={"feedback_id": feedback_id}
        )
    except Exception as e:
        logger.error(f"提交反馈失败：{e}")
        return APIResponse(
            success=False,
            message="提交失败",
            error=str(e)
        )

@router.get("/feedback/list", tags=["反馈"])
async def list_feedbacks():
    """获取反馈列表"""
    feedback_dir = Path("./feedback_data")
    if not feedback_dir.exists():
        return []
    
    feedbacks = []
    for f in sorted(feedback_dir.glob("*.json"), reverse=True):
        with open(f, 'r', encoding='utf-8') as file:
            feedbacks.append(json.load(file))
    
    return feedbacks