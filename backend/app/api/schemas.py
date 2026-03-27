# backend/app/api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ==================== 请求模型 ====================
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=[], description="对话历史")
    retrieval_mode: str = Field(default="hybrid_rerank", description="检索模式")
    top_k: int = Field(default=5, ge=1, le=20, description="检索数量")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="元数据过滤")
   # ==================== 新增医疗场景字段 ====================
    department: Optional[str] = Field(None, description="科室")
    urgency: bool = Field(default=False, description="是否紧急")
    enable_safety_check: bool = Field(default=True, description="启用安全检查")
    enable_citation: bool = Field(default=True, description="启用引用标注")
    stream: bool = Field(default=False, description="是否流式输出")

class StreamChatRequest(ChatRequest):
    """流式聊天请求（新增）"""
    stream: bool = Field(default=True, description="流式输出")
 # ====================  评估相关请求 ====================   
class EvaluationConfig(BaseModel):
    """评估配置（新增）"""
    retrieval_mode: Optional[str] = Field(None, description="检索模式")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="检索数量")
    metrics: Optional[List[str]] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision"],
        description="评估指标"
    )
    enable_medical_metrics: bool = Field(default=True, description="启用医疗指标")

class EvaluateSingleRequest(BaseModel):
    """单问题评估请求（新增）"""
    question_id: str = Field(..., min_length=1, description="问题 ID")
    config: Optional[EvaluationConfig] = Field(None, description="评估配置")


class EvaluateBatchRequest(BaseModel):
    """批量评估请求（新增）"""
    dataset_name: str = Field(..., min_length=1, description="测试数据集名称")
    config: Optional[EvaluationConfig] = Field(None, description="评估配置")
    sample_rate: float = Field(default=1.0, ge=0.01, le=1.0, description="采样率")
    callback_url: Optional[str] = Field(None, description="回调 URL")


class CompareModesRequest(BaseModel):
    """模式对比请求（新增）"""
    modes: List[str] = Field(..., min_length=2, max_length=5, description="要对比的模式")
    dataset_name: Optional[str] = Field(None, description="数据集名称")

class DocumentUploadRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="文档列表")
    collection_name: Optional[str] = Field(default=None, description="集合名称")

class DocumentDeleteRequest(BaseModel): 
    ids: List[str] = Field(..., description="文档 ID 列表")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="元数据过滤")

# === 向量库管理请求（扩展）===

class DocumentMetadata(BaseModel):
    """文档元数据（新增）"""
    source: str = Field(..., description="来源文件")
    department: Optional[str] = Field(None, description="科室")
    category: Optional[str] = Field(None, description="分类")
    page: Optional[int] = Field(None, ge=1, description="页码")
    publish_date: Optional[str] = Field(None, description="发布日期")
    custom: Optional[Dict[str, Any]] = Field(None, description="自定义字段")

class DocumentContent(BaseModel):
    """文档内容（新增）"""
    content: str = Field(..., min_length=1, max_length=10000, description="文档内容")
    metadata: DocumentMetadata = Field(..., description="元数据")
class DocumentUploadRequest(BaseModel):
    """文档上传请求（扩展）"""
    documents: List[Dict[str, Any]] = Field(..., description="文档列表")
    collection_name: Optional[str] = Field(default=None, description="集合名称")
    # 新增字段（可选）
    auto_chunk: bool = Field(default=True, description="自动分块")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="分块大小")


class DocumentDeleteRequest(BaseModel):
    """文档删除请求（保持兼容）"""
    ids: List[str] = Field(..., description="文档 ID 列表")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="元数据过滤")

# === 用户反馈请求（新增）===
class FeedbackRequest(BaseModel):
    """用户反馈请求（新增）"""
    query: str = Field(..., min_length=1, description="原始问题")
    answer_id: Optional[str] = Field(None, description="回答 ID")
    rating: Optional[int] = Field(None, ge=1, le=5, description="评分 1-5")
    comment: Optional[str] = Field(None, max_length=1000, description="详细意见")
    issue_type: Optional[str] = Field(None, description="问题类型")


# ==================== 响应模型 ====================
class CitationInfo(BaseModel):
    index: int
    source: str
    page: Optional[str]
    content: str
    score: Optional[float]
    department: Optional[str] = Field(None, description="科室")
    evidence_level: Optional[str] = Field(None, description="证据等级")

class QualityAssessment(BaseModel):
    has_citation: bool
    has_disclaimer: bool
    confidence: str
    warnings: List[str]
    completeness: Optional[float] = Field(None, ge=0, le=1, description="完整度")
    safety_score: Optional[float] = Field(None, ge=0, le=1, description="安全得分")

class SafetyCheckResult(BaseModel):
    """安全检查结果（新增）"""
    passed: bool
    flags: List[str] = []
    warnings: List[str] = []
    emergency_detected: bool = False
    recommendation: Optional[str] = None


class RetrievalStats(BaseModel):
    """检索统计（新增）"""
    mode: str
    total_retrieved: int
    after_rerank: int
    retrieval_time: float

class ChatResponse(BaseModel):
    success: bool
    query: str
    answer: str
    citations: List[CitationInfo]
    quality: QualityAssessment
    stats: Dict[str, Any]
    timestamp: datetime
    error: Optional[str] = None
    safety: Optional[SafetyCheckResult] = Field(None, description="安全检查")
    retrieval: Optional[RetrievalStats] = Field(None, description="检索统计")
    conversation_id: Optional[str] = Field(None, description="会话 ID")

class StreamChunk(BaseModel):
    """流式数据块（新增）"""
    type: str = Field(..., description="数据类型", examples=["chunk", "citation", "complete", "error"])
    content: Optional[str] = Field(None, description="文本内容")
    citation: Optional[CitationInfo] = Field(None, description="引用信息")
    stats: Optional[Dict[str, Any]] = Field(None, description="统计信息")
    error: Optional[str] = Field(None, description="错误信息")

# === 评估响应（新增）===

class MetricScore(BaseModel):
    """指标得分（新增）"""
    name: str
    score: float
    description: Optional[str] = None


class EvaluationAggregateStats(BaseModel):
    """评估聚合统计（新增）"""
    retrieval: Dict[str, float] = {}
    generation: Dict[str, float] = {}
    safety: Dict[str, float] = {}
    overall: Dict[str, float] = {}


class IndividualEvaluationResult(BaseModel):
    """单问题评估结果（新增）"""
    question_id: str
    query: str
    metrics: List[MetricScore] = []
    overall_score: float
    issues: List[str] = []


class EvaluationReport(BaseModel):
    """评估报告（新增）"""
    evaluation_id: str
    status: str = Field(..., examples=["running", "completed", "failed"])
    config: Dict[str, Any] = {}
    aggregate_stats: Optional[EvaluationAggregateStats] = None
    individual_results: List[IndividualEvaluationResult] = []
    recommendations: List[str] = []
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_time: Optional[float] = None


class EvaluationResponse(BaseModel):
    """评估响应（新增）"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class CompareModesResponse(BaseModel):
    """模式对比响应（新增）"""
    success: bool
    comparison: Dict[str, Dict[str, float]] = {}
    best_mode: Optional[str] = None
    recommendation: Optional[str] = None

# === 向量库响应（扩展）===

class DocumentStats(BaseModel):
    """文档统计（新增）"""
    total_count: int
    by_department: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    last_updated: Optional[datetime] = None


class UploadResult(BaseModel):
    """上传结果（新增）"""
    total_submitted: int
    successfully_added: int
    duplicates_skipped: int
    failed: int = 0
    errors: List[Dict[str, Any]] = []


class UploadDocumentsResponse(BaseModel):
    """文档上传响应（新增）"""
    success: bool
    message: str
    data: Optional[UploadResult] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class DeleteResult(BaseModel):
    """删除结果（新增）"""
    deleted_count: int
    affected_ids: List[str] = []


class DeleteDocumentsResponse(BaseModel):
    """文档删除响应（新增）"""
    success: bool
    message: str
    data: Optional[DeleteResult] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)



class CollectionStats(BaseModel):
    collection_name: str
    total_documents: int
    last_updated: datetime
    embedding_model: Optional[str] = None
    health: Optional[str] = Field("healthy", examples=["healthy", "degraded", "error"])

class CollectionInfo(BaseModel):
    """集合信息（新增）"""
    name: str
    total_documents: int
    embedding_model: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    health: str = "healthy"

class CollectionStatsResponse(BaseModel):
    """集合统计响应（新增）"""
    success: bool
    message: str
    data: Optional[CollectionInfo] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# === 系统响应（保持兼容 + 扩展）===
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, bool]
    details: Optional[Dict[str, Any]] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

class SystemInfo(BaseModel):
    """系统信息（新增）"""
    app_name: str
    version: str
    debug: bool
    embedding_model: str
    rerank_model: str
    default_top_k: int


# === 反馈响应（新增）===

class FeedbackResponse(BaseModel):
    """反馈响应（新增）"""
    success: bool
    message: str
    feedback_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)