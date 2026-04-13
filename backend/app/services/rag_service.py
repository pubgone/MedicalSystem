# backend/app/services/rag_service.py
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from FlagEmbedding import FlagReranker
import os
import asyncio
from ..core.config import settings
from ..utils.logger import get_logger
from pathlib import Path


logger = get_logger(__name__)

class RAGService:
    """RAG 业务服务层"""
    _instance = None  # 单例实例
    
    def __new__(cls):
        """确保单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):

        if hasattr(self, '_initialized') and self._initialized:
            logger.info("⚠️  RAGService 已初始化，跳过重复加载")
            return
        logger.info("初始化 RAG 服务...")

        try:
            # 1. 初始化 Embeddings
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': settings.DEVICE, 'local_files_only': True},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 2. 初始化向量库 ✅ 确保属性名是 vector_store
            self.vector_store = Chroma(
                collection_name=settings.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
            
            # 3. 初始化 Rerank
            self.reranker = FlagReranker(settings.RERANK_MODEL, use_fp16=True)
            
            # 4. 初始化检索器
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from rag.retrievers import MedicalRetriever
            
            self.retriever = MedicalRetriever(
                vector_store=self.vector_store,  # ✅ 使用 self.vector_store
                embedding_function=self.embeddings,
                rerank_model_name=settings.RERANK_MODEL,
                top_k=settings.DEFAULT_TOP_K * 2,
                rerank_top_k=settings.DEFAULT_RERANK_TOP_K,
                show_progress=False
            )
            
            # 5. 初始化 LLM
            self.llm = self._init_llm()
            
            # 6. 初始化 Prompt
            from config.prompt_config import create_medical_prompt
            self.prompt = create_medical_prompt()
            
            # ✅ 标记已初始化
            self._initialized = True
            logger.info("✅ RAG 服务初始化完成")
            
        except Exception as e:
            logger.error(f"❌ RAG 服务初始化失败：{e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self._initialized = False
            raise
    
    def _init_llm(self):
        """初始化 LLM"""
        # 根据实际使用的模型调整
        try:
            logger.info(f"🔄 初始化 LLM: endpoint={settings.LLM_ENDPOINT}")

            # 检查 API Key
            api_key = settings.LLM_KEY or os.getenv("LLM_KEY")
            if not api_key:
                logger.warning("⚠️  未配置 DASHSCOPE_API_KEY，使用 Mock LLM")
                from langchain_community.llms import FakeListLLM
                return FakeListLLM(responses=[
                    "⚠️ 测试模式：请配置 DASHSCOPE_API_KEY 环境变量以使用真实 LLM。"
                ])        
            from langchain_qwq import ChatQwen
            llm = ChatQwen(
                base_url=settings.LLM_ENDPOINT,
                api_key=settings.LLM_KEY,
                model=settings.LLM_NAME,
                max_tokens=3_000,
                timeout=None,
                max_retries=2,
            )
            # 测试连接
            logger.info("🔄 测试 LLM 连接...")
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([("user", "test")])
            chain = prompt | llm
            chain.invoke({})

            logger.info("✅ LLM 初始化成功")
            return llm

        except Exception as e:
            logger.error(f"❌ LLM 初始化失败：{e}")
            # 返回 Mock LLM 避免服务崩溃
            from langchain_community.llms import FakeListLLM
            return FakeListLLM(responses=[f"⚠️ LLM 初始化失败：{e}"])    
    ###############################检查测试#################################################
    def check_llm_health(self) -> Dict[str, Any]:
        """检查 LLM 健康状态 - 带详细日志"""
        try:
            logger.info("🔍 开始检查 LLM 连接...")
            logger.info(f"LLM 类型：{type(self.llm).__name__}")
            
            # 打印配置（脱敏）
            if hasattr(self.llm, 'openai_api_base'):
                logger.info(f"API Base: {self.llm.openai_api_base[:50]}...")
            if hasattr(self.llm, 'model_name'):
                logger.info(f"Model: {self.llm.model_name}")
            
            # 简单测试调用
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([("user", "Hi")])
            chain = prompt | self.llm
            
            logger.info("🔄 发送测试请求...")
            response = chain.invoke({})
            logger.info(f"✅ LLM 响应成功：{str(response)[:100]}")
            
            return {"healthy": True, "message": "OK"}
            
        except Exception as e:
            logger.error(f"❌ LLM 健康检查失败：{type(e).__name__}: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {"healthy": False, "error": str(e), "error_type": type(e).__name__}    
    ###############################检查测试#################################################
    def chat(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieval_mode: str = "hybrid_rerank",
        top_k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行 RAG 问答"""
        top_k = top_k or settings.DEFAULT_TOP_K

        if filter is not None and len(filter) == 0:
           filter = None     

        # 1. 检索
        search_result = self.retriever.search(
            query=query,
            mode=retrieval_mode,
            k=top_k * 2,
            rerank_k=top_k,
            filter=filter,
            show_time=False
        )
        
        documents = search_result["documents"]
        
        # 2. 构建上下文
        context = self._format_context(documents)
        
        # 3. 构建 Prompt
        chat_history_str = self._format_history(chat_history)
        inputs = {
            "context": context,
            "question": query,
            "chat_history": chat_history_str
        }
        
        # 4. 调用 LLM
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(inputs)
        
        # 5. 构建返回结果
        return {
            "answer": answer,
            "documents": documents,
            "citations": self._extract_citations(documents, search_result.get("scores", [])),
            "quality": self._assess_quality(answer, documents),
            "stats": search_result.get("stats", {})
        }
    
    async def chat_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieval_mode: str = "hybrid_rerank",
        top_k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式 RAG 问答"""
        top_k = top_k or settings.DEFAULT_TOP_K

        if filter is not None and len(filter) == 0:
           filter = None     
                     
        # 1. 先检索（同步）
        search_result = self.retriever.search(
            query=query,
            mode=retrieval_mode,
            k=top_k * 2,
            rerank_k=top_k,
            filter=filter,
            show_time=False
        )
        
        documents = search_result["documents"]
        context = self._format_context(documents)
        
        # 2. 发送检索完成事件
        yield {
            "type": "retrieval_complete",
            "document_count": len(documents),
            "stats": search_result.get("stats", {})
        }
        
        # 3. 流式生成回答
        chat_history_str = self._format_history(chat_history)
        inputs = {
            "context": context,
            "question": query,
            "chat_history": chat_history_str
        }
        
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer_chunks = []
        for chunk in chain.stream(inputs):
            answer_chunks.append(chunk)
            yield {
                "type": "chunk",
                "content": chunk
            }
        
        # 4. 发送完成事件
        yield {
            "type": "complete",
            "answer": "".join(answer_chunks),
            "citations": self._extract_citations(documents, search_result.get("scores", []))
        }
    
    def _format_context(self, documents: List) -> str:
        """格式化检索结果为上下文"""
        if not documents:
            return "暂无相关医疗文献资料。"
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content[:500]
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            
            context_part = f"""[文献{i}]
来源：{source}{' (第' + str(page) + '页)' if page else ''}
内容：{content}
---"""
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)[:settings.MAX_CONTEXT_LENGTH]
    
    def _format_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """格式化对话历史"""
        if not chat_history:
            return ""
        
        history_str = ""
        for msg in chat_history[-5:]:
            role = "用户" if msg["role"] == "human" else "助手"
            history_str += f"{role}: {msg['content']}\n"
        return history_str
    
    def _extract_citations(self, documents: List, scores: List[float]) -> List[Dict[str, Any]]:
        """提取引用信息"""
        citations = []
        for i, doc in enumerate(documents, 1):
            citations.append({
                "index": i,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", ""),
                "content": doc.page_content[:200],
                "score": scores[i-1] if i <= len(scores) else None
            })
        return citations
    
    def _assess_quality(self, answer: str, documents: List) -> Dict[str, Any]:
        """评估回答质量"""
        return {
            "has_citation": "[文献" in answer or "来源：" in answer,
            "has_disclaimer": "免责" in answer or "咨询医生" in answer,
            "confidence": "high" if len(documents) >= 3 else "medium",
            "warnings": [] if len(documents) >= 3 else ["文献支持较少"]
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取向量库统计"""
        try:
            collection = self.vector_store._client.get_collection(
                self.vector_store._collection.name
            )
            return {
                "collection_name": settings.COLLECTION_NAME,
                "total_documents": collection.count()
            }
        except Exception as e:
            logger.error(f"获取统计失败：{e}")
            return {"collection_name": settings.COLLECTION_NAME, "total_documents": 0}
    
    def upload_documents(self, documents: List[Dict], collection_name: str = None) -> int:
        """上传文档"""
        from langchain_core.documents import Document
        docs = [Document(page_content=d["content"], metadata=d.get("metadata", {})) for d in documents]
        ids = self.retriever.vector_store.add_documents(docs)
        self.retriever.refresh_bm25_index()
        return len(ids)
    
    def delete_documents(self, ids: List[str] = None, filter: Dict = None) -> int:
        """删除文档"""
        if ids:
            self.retriever.vector_store.delete(ids=ids)
            self.retriever.refresh_bm25_index()
            return len(ids)
        return 0
    
    def check_llm_health(self) -> bool:
        """检查 LLM 健康状态"""
        try:
            # 简单测试
            self.llm.invoke("测试")
            return True
        except:
            return False