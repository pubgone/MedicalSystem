# medical_chain.py
import time
from typing import List, Dict, Any, Optional, Generator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from config.prompt_config import (
    create_medical_prompt,
    INSUFFICIENT_INFO_TEMPLATE,
    check_emergency,
    EMERGENCY_RESPONSE
)

class MedicalRAGChain:
    """
    医疗 RAG 链式调用封装
    支持：流式输出、引用标注、置信度评估、安全检测
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        retriever: Any,  # MedicalRetriever 实例
        prompt: Optional[PromptTemplate] = None,
        max_context_length: int = 4000,
        enable_safety_check: bool = True,
        enable_citation: bool = True,
        verbose: bool = True
    ):
        """
        :param llm: 大语言模型实例
        :param retriever: 检索器实例
        :param prompt: 自定义 Prompt 模板
        :param max_context_length: 最大上下文长度
        :param enable_safety_check: 启用安全检查
        :param enable_citation: 启用引用标注
        :param verbose: 是否输出详细信息
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt or create_medical_prompt()
        self.max_context_length = max_context_length
        self.enable_safety_check = enable_safety_check
        self.enable_citation = enable_citation
        self.verbose = verbose
        
        # 构建 Chain
        self.chain = self._build_chain()
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "total_time": 0,
            "emergency_detected": 0,
            "insufficient_info": 0
        }
    
    def _build_chain(self):
        """构建 RAG Chain"""
        # 使用 RunnablePassthrough 传递检索结果
        rag_chain = (
            RunnablePassthrough.assign(context=self._format_context)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    
    def _format_context(self, inputs: Dict[str, Any]) -> str:
        """格式化检索结果为上下文"""
        documents = inputs.get("documents", [])
        
        if not documents:
            return "暂无相关医疗文献资料。"
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # 截取内容避免过长
            content = doc.page_content
            if len(content) > 500:
                content = content[:500] + "..."
            
            # 添加来源信息
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            
            context_part = f"""[文献{i}]
来源：{source}{' (第' + str(page) + '页)' if page else ''}
内容：{content}
---"""
            context_parts.append(context_part)
        
        full_context = "\n\n".join(context_parts)
        
        # 检查是否超过最大长度
        if len(full_context) > self.max_context_length:
            full_context = full_context[:self.max_context_length] + "\n\n[内容已截断]"
        
        return full_context
    
    def _check_response_quality(self, response: str, documents: List[Document]) -> Dict[str, Any]:
        """评估回答质量"""
        quality = {
            "has_citation": False,
            "has_disclaimer": False,
            "confidence": "medium",
            "warnings": []
        }
        
        # 检查是否有引用
        if self.enable_citation and ("[文献" in response or "来源：" in response):
            quality["has_citation"] = True
        
        # 检查是否有免责声明
        if "免责" in response or "参考" in response or "咨询医生" in response:
            quality["has_disclaimer"] = True
        
        # 检查是否表示信息不足
        if "无法" in response or "不足" in response or "抱歉" in response:
            quality["confidence"] = "low"
            quality["warnings"].append("信息可能不完整")
        
        # 检查文档数量
        if len(documents) == 0:
            quality["confidence"] = "low"
            quality["warnings"].append("无相关文献支持")
        elif len(documents) < 3:
            quality["confidence"] = "medium"
            quality["warnings"].append("文献支持较少")
        else:
            quality["confidence"] = "high"
        
        return quality
    
    def invoke(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieval_mode: str = "hybrid_rerank",
        top_k: int = 5,
        **retrieval_kwargs
    ) -> Dict[str, Any]:
        """
        完整 RAG 调用
        :return: 包含回答、引用、统计信息的字典
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        result = {
            "query": query,
            "answer": "",
            "documents": [],
            "citations": [],
            "quality": {},
            "stats": {},
            "error": None
        }
        
        try:
            # 1. 安全检查
            if self.enable_safety_check and check_emergency(query):
                self.stats["emergency_detected"] += 1
                result["answer"] = EMERGENCY_RESPONSE
                result["stats"]["emergency"] = True
                return result
            
            # 2. 检索相关文档
            if self.verbose:
                print(f"🔍 正在检索医疗文献...")
            
            search_result = self.retriever.search(
                query=query,
                mode=retrieval_mode,
                k=top_k * 2,  # 检索更多供 LLM 筛选
                show_time=self.verbose,
                **retrieval_kwargs
            )
            
            documents = search_result["documents"]
            result["documents"] = documents
            
            # 3. 检查是否有足够信息
            if not documents or len(documents) == 0:
                self.stats["insufficient_info"] += 1
                result["answer"] = INSUFFICIENT_INFO_TEMPLATE
                result["quality"]["confidence"] = "low"
                return result
            
            # 4. 构建输入
            chat_history_str = ""
            if chat_history:
                for msg in chat_history[-5:]:  # 只保留最近 5 轮
                    role = "用户" if msg["role"] == "human" else "助手"
                    chat_history_str += f"{role}: {msg['content']}\n"
            
            inputs = {
                "question": query,
                "documents": documents,
                "chat_history": chat_history_str
            }
            
            # 5. 调用 Chain
            if self.verbose:
                print(f"🤖 正在生成回答...")
            
            response = self.chain.invoke(inputs)
            result["answer"] = response
            
            # 6. 提取引用
            if self.enable_citation:
                result["citations"] = self._extract_citations(documents, response)
            
            # 7. 质量评估
            result["quality"] = self._check_response_quality(response, documents)
            
        except Exception as e:
            result["error"] = str(e)
            result["answer"] = f"⚠️ 生成回答时出现错误：{str(e)}\n\n建议您重新提问或咨询专业医生。"
        
        # 统计时间
        elapsed = time.time() - start_time
        result["stats"]["total_time"] = elapsed
        result["stats"]["retrieval_time"] = search_result.get("stats", {}).get("total_time", 0) if 'search_result' in locals() else 0
        self.stats["total_time"] += elapsed
        
        if self.verbose:
            print(f"⏱️  总耗时：{elapsed:.2f}s")
        
        return result
    
    def stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieval_mode: str = "hybrid_rerank",
        top_k: int = 5,
        **retrieval_kwargs
    ) -> Generator[str, None, None]:
        """
        流式输出 RAG 回答
        :yield: 生成的文本片段
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        try:
            # 1. 安全检查
            if self.enable_safety_check and check_emergency(query):
                self.stats["emergency_detected"] += 1
                yield EMERGENCY_RESPONSE
                return
            
            # 2. 检索
            if self.verbose:
                print(f"🔍 正在检索...", flush=True)
            
            search_result = self.retriever.search(
                query=query,
                mode=retrieval_mode,
                k=top_k * 2,
                show_time=False,
                **retrieval_kwargs
            )
            
            documents = search_result["documents"]
            
            # 3. 信息不足
            if not documents:
                self.stats["insufficient_info"] += 1
                yield INSUFFICIENT_INFO_TEMPLATE
                return
            
            # 4. 构建输入
            chat_history_str = ""
            if chat_history:
                for msg in chat_history[-5:]:
                    role = "用户" if msg["role"] == "human" else "助手"
                    chat_history_str += f"{role}: {msg['content']}\n"
            
            inputs = {
                "question": query,
                "documents": documents,
                "chat_history": chat_history_str
            }
            
            # 5. 流式生成
            if self.verbose:
                print(f"🤖 正在生成...", flush=True)
            
            for chunk in self.chain.stream(inputs):
                yield chunk
        
        except Exception as e:
            yield f"\n\n⚠️ 错误：{str(e)}"
    
    def _extract_citations(self, documents: List[Document], response: str) -> List[Dict[str, Any]]:
        """从回答中提取引用信息"""
        citations = []
        for i, doc in enumerate(documents, 1):
            citation = {
                "index": i,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", ""),
                "content": doc.page_content[:200],
                "score": None  # 如果有 Rerank 分数可以填入
            }
            citations.append(citation)
        return citations
    
    def get_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            **self.stats,
            "avg_time": self.stats["total_time"] / self.stats["total_queries"] 
                       if self.stats["total_queries"] > 0 else 0
        }
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_queries": 0,
            "total_time": 0,
            "emergency_detected": 0,
            "insufficient_info": 0
        }