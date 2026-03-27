# retrieval_module.py
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from config.config import config

class MedicalRetriever:
    """
    医疗知识检索模块
    支持：普通向量检索、混合检索、Rerank 重排序
    """
    
    def __init__(
        self,
        vector_store: Chroma,
        embedding_function: Embeddings,
        rerank_model_name: str = config.RERANK_MODEL,  # 中文优化
        top_k: int = 10,
        rerank_top_k: int = 5,
        show_progress: bool = True
    ):
        """
        :param vector_store: Chroma 向量库实例
        :param embedding_function: 嵌入模型
        :param rerank_model_name: Rerank 模型名称
        :param top_k: 初始检索数量
        :param rerank_top_k: Rerank 后返回数量
        :param show_progress: 是否显示进度
        """
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.show_progress = show_progress
        
        # 初始化 Rerank 模型
        if self.show_progress:
            print(f"🔄 加载 Rerank 模型：{rerank_model_name}")
        self.reranker = FlagReranker(rerank_model_name, use_fp16=True)
        if self.show_progress:
            print("✅ Rerank 模型加载完成")
        
        # BM25 索引（用于混合检索）
        self.bm25_retriever = None
        self._bm25_indexed = False
    
    def _build_bm25_index(self, force_rebuild: bool = False, batch_size: int = 500):
        """构建 BM25 关键词索引（分批获取避免 SQL 限制）"""
        if self._bm25_indexed and not force_rebuild:
            return
        
        if self.show_progress:
            print("📚 构建 BM25 索引中...")
        
        all_documents = []
        
        try:
            collection = self.vector_store._client.get_collection(
                self.vector_store._collection.name
            )
            
            # 获取总数
            total_count = collection.count()
            
            # 分批获取
            with tqdm(
                total=total_count, 
                desc="📖 加载文档", 
                unit="doc",
                disable=not self.show_progress
            ) as pbar:
                for offset in range(0, total_count, batch_size):
                    batch = collection.get(
                        include=["documents", "metadatas"],
                        limit=batch_size,
                        offset=offset
                    )
                    
                    docs = [
                        Document(page_content=doc, metadata=meta)
                        for doc, meta in zip(batch['documents'], batch['metadatas'])
                    ]
                    all_documents.extend(docs)
                    pbar.update(len(docs))
            
            # 创建 BM25 检索器
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=all_documents,
                k=self.top_k
            )
            self._bm25_indexed = True
            
            if self.show_progress:
                print(f"✅ BM25 索引构建完成，共 {len(all_documents)} 条文档")
                
        except Exception as e:
            print(f"❌ BM25 索引构建失败：{e}")
            self._bm25_indexed = False
    
    def refresh_bm25_index(self):
        """刷新 BM25 索引（向量库更新后调用）"""
        self._bm25_indexed = False
        self._build_bm25_index(force_rebuild=True)
    
    # ==================== 普通向量检索 ====================
    def vector_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        show_time: bool = True
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        普通向量相似度检索
        :return: (文档列表，检索统计信息)
        """
        k = k or self.top_k
        stats = {"method": "vector", "query": query}
        
        if filter is not None and len(filter) == 0:
          filter = None      
                  
        start_time = time.time()
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        elapsed = time.time() - start_time
        
        stats["elapsed_time"] = elapsed
        stats["results_count"] = len(results)
        stats["speed"] = len(results) / elapsed if elapsed > 0 else 0
        
        if show_time and self.show_progress:
            print(f"⏱️  向量检索：{elapsed:.3f}s, 返回 {len(results)} 条结果")
        
        return results, stats
    
    # ==================== 混合检索 ====================
    def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        show_time: bool = True
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        混合检索：向量 + BM25 关键词，加权融合
        :param vector_weight: 向量检索权重
        :param bm25_weight: BM25 权重
        :return: (文档列表，检索统计信息)
        """
        k = k or self.top_k
        stats = {"method": "hybrid", "query": query}

        if filter is not None and len(filter) == 0:
          filter = None      

        # 确保 BM25 索引已构建
        if not self._bm25_indexed:
            self._build_bm25_index()
        
        start_time = time.time()
        
        # 1. 向量检索
        vector_results, _ = self.vector_search(query, k=k*2, filter=filter, show_time=False)
        
        # 2. BM25 检索
        bm25_results = []
        if self.bm25_retriever:
            try:
                bm25_results = self.bm25_retriever.invoke(query, k=k*2)
            except Exception as e:
                if self.show_progress:
                    print(f"⚠️  BM25 检索失败：{e}")
        
        # 3. 结果融合（基于文档内容去重 + 加权评分）
        merged_results = self._merge_results(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=k
        )
        
        elapsed = time.time() - start_time
        stats["elapsed_time"] = elapsed
        stats["results_count"] = len(merged_results)
        stats["vector_count"] = len(vector_results)
        stats["bm25_count"] = len(bm25_results)
        
        if show_time and self.show_progress:
            print(f"⏱️  混合检索：{elapsed:.3f}s, 返回 {len(merged_results)} 条结果")
        
        return merged_results, stats
    
    def _merge_results(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
        vector_weight: float,
        bm25_weight: float,
        top_k: int
    ) -> List[Document]:
        """融合向量检索和 BM25 检索结果"""
        # 使用文档内容作为唯一键
        doc_scores = {}
        doc_map = {}
        
        # 向量检索分数（归一化）
        for i, doc in enumerate(vector_results):
            score = (top_k - i) / top_k  # 位置越前分数越高
            content_key = doc.page_content[:100]  # 用前 100 字作为键
            if content_key not in doc_scores:
                doc_scores[content_key] = 0
                doc_map[content_key] = doc
            doc_scores[content_key] += score * vector_weight
        
        # BM25 检索分数
        for i, doc in enumerate(bm25_results):
            score = (top_k - i) / top_k
            content_key = doc.page_content[:100]
            if content_key not in doc_scores:
                doc_scores[content_key] = 0
                doc_map[content_key] = doc
            doc_scores[content_key] += score * bm25_weight
        
        # 按分数排序
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [doc_map[key] for key, _ in sorted_docs]
    
    # ==================== Rerank 重排序 ====================
    def rerank_results(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        show_time: bool = True
    ) -> Tuple[List[Document], List[float], Dict[str, Any]]:
        """
        使用交叉编码器对检索结果重排序
        :return: (重排序后的文档，对应分数，统计信息)
        """
        top_k = top_k or self.rerank_top_k
        stats = {"method": "rerank", "query": query}
        
        if not documents:
            return [], [], stats
        
        start_time = time.time()
        
        # 准备 Rerank 输入
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 执行 Rerank
        scores = self.reranker.compute_score(pairs, normalize=True)
        
        # 如果是单个分数，转为列表
        if isinstance(scores, float):
            scores = [scores]
        
        # 按分数排序
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 取前 top_k
        ranked_docs = [doc for doc, _ in doc_score_pairs[:top_k]]
        ranked_scores = [score for _, score in doc_score_pairs[:top_k]]
        
        elapsed = time.time() - start_time
        stats["elapsed_time"] = elapsed
        stats["original_count"] = len(documents)
        stats["reranked_count"] = len(ranked_docs)
        
        if show_time and self.show_progress:
            print(f"⏱️  Rerank：{elapsed:.3f}s, {len(documents)}→{len(ranked_docs)} 条")
        
        return ranked_docs, ranked_scores, stats
    
    # ==================== 完整检索流程 ====================
    def search(
        self,
        query: str,
        mode: str = "hybrid_rerank",  # vector | hybrid | hybrid_rerank
        k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        show_time: bool = True
    ) -> Dict[str, Any]:
        """
        统一检索接口
        :param mode: 检索模式
            - vector: 普通向量检索
            - hybrid: 混合检索
            - hybrid_rerank: 混合检索 + Rerank
        :return: 包含文档、分数、统计信息的字典
        """
        k = k or self.top_k
        rerank_k = rerank_k or self.rerank_top_k
        if filter is not None and len(filter) == 0:
            filter = None

        if self.show_progress:
            print(f"\n🔍 检索模式：{mode}")
            print(f"📝 查询：{query[:50]}..." if len(query) > 50 else f"📝 查询：{query}")        

        
        total_start = time.time()
        result = {
            "query": query,
            "mode": mode,
            "documents": [],
            "scores": [],
            "stats": {}
        }
        
        # 1. 基础检索
        if mode == "vector":
            docs, stats = self.vector_search(query, k=k, filter=filter, show_time=show_time)
            result["documents"] = docs
            result["stats"]["retrieval"] = stats
            
        elif mode == "hybrid":
            docs, stats = self.hybrid_search(
                query, k=k, filter=filter,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                show_time=show_time
            )
            result["documents"] = docs
            result["stats"]["retrieval"] = stats
            
        elif mode == "hybrid_rerank":
            # 混合检索
            docs, stats = self.hybrid_search(
                query, k=k*2, filter=filter,  # 检索更多供 Rerank 筛选
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                show_time=show_time
            )
            result["stats"]["retrieval"] = stats
            
            # Rerank
            ranked_docs, scores, rerank_stats = self.rerank_results(
                query, docs, top_k=rerank_k, show_time=show_time
            )
            result["documents"] = ranked_docs
            result["scores"] = scores
            result["stats"]["rerank"] = rerank_stats
        else:
            raise ValueError(f"不支持的检索模式：{mode}")
        
        total_elapsed = time.time() - total_start
        result["stats"]["total_time"] = total_elapsed
        
        if show_time and self.show_progress:
            print(f"⏱️  总耗时：{total_elapsed:.3f}s")
            print(f"📄 最终返回：{len(result['documents'])} 条文档\n")
        
        return result
    
    # ==================== 结果格式化 ====================
    def format_results(
        self,
        search_result: Dict[str, Any],
        show_score: bool = True,
        show_metadata: bool = True
    ) -> str:
        """格式化检索结果为可读文本"""
        output = []
        output.append("=" * 60)
        output.append(f"🔍 查询：{search_result['query']}")
        output.append(f"📊 模式：{search_result['mode']}")
        output.append(f"⏱️  总耗时：{search_result['stats'].get('total_time', 0):.3f}s")
        output.append("=" * 60)
        
        for i, doc in enumerate(search_result['documents'], 1):
            output.append(f"\n【{i}】")
            output.append(f"内容：{doc.page_content[:200]}..." if len(doc.page_content) > 200 else f"内容：{doc.page_content}")
            
            if show_score and search_result.get('scores'):
                output.append(f"分数：{search_result['scores'][i-1]:.4f}")
            
            if show_metadata and doc.metadata:
                meta_str = ", ".join([f"{k}={v}" for k, v in doc.metadata.items()])
                output.append(f"元数据：{meta_str}")
        
        output.append("\n" + "=" * 60)
        return "\n".join(output)