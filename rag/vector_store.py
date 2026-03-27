import hashlib
import uuid
from typing import List, Optional, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config.config import config
import chromadb
from tqdm import tqdm

class VectorStore:
    def __init__(
        self, 
        collection_name: str, 
        persist_directory: str = config.CHROMA_PERSIST_DIR, 
        embedding_model_name: str = config.EMBEDDING_MODEL,
        device: str = "cpu",
        show_progress: bool = True  # 控制是否显示进度条
    ):
        """
        初始化医疗向量库
        :param collection_name: 集合名称 (例如：medical_guidelines_v1)
        :param embedding_function: 嵌入模型 (例如：HuggingFaceEmbeddings)
        :param persist_directory: 本地持久化路径
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.device = device
        self.show_progress = show_progress
        
        # 1. 初始化嵌入模型
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device, 'local_files_only': True },
            encode_kwargs={'normalize_embeddings': True}
        )
                # 初始化 Chroma 客户端
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
    def _generate_unique_id(self, document: Document) -> str:
        """
        【核心优化】动态生成确定性唯一 ID
        策略：按优先级提取 metadata 中的字段参与哈希
        顺序：source -> page -> section (如果有) -> content
        这样即使未来添加了 page 字段，ID 也会自动变化以区分版本
        """
        content = document.page_content.strip()
        meta = document.metadata
        
        # 构建哈希的关键部分列表
        hash_parts = []
        
        # 1. 必须包含：来源
        source = meta.get("source", "unknown_source")
        hash_parts.append(f"src:{source}")
        
        # 2. 可选包含：页码 (如果有)
        if "page" in meta and meta["page"] is not None:
            hash_parts.append(f"page:{meta['page']}")
            
        # 3. 可选包含：章节/段落 (如果有，未来扩展用)
        if "section" in meta and meta["section"] is not None:
            hash_parts.append(f"sec:{meta['section']}")
            
        # 4. 必须包含：内容本身
        hash_parts.append(f"content:{content}")
        
        # 组合成字符串并生成 MD5 哈希
        unique_str = "|".join(hash_parts)
        doc_id = hashlib.md5(unique_str.encode('utf-8')).hexdigest()
        
        return doc_id
    
    def _deduplicate_by_id(self, documents: List[Document], ids: List[str]) -> Tuple[List[Document], List[str], int]:
        """
        【关键修复】在批次内去重
        如果多个文档生成了相同的 ID，只保留最后一个（视为最新版本）
        返回：去重后的文档列表、ID 列表、被移除的重复数量
        """
        seen_ids = {}
        for i, doc_id in enumerate(ids):
            # 字典赋值会覆盖旧值，自然实现"保留最后一个"
            seen_ids[doc_id] = i
            
        # 获取保留下来的索引
        unique_indices = sorted(seen_ids.values())
        
        unique_docs = [documents[i] for i in unique_indices]
        unique_ids = [ids[i] for i in unique_indices]
        
        removed_count = len(documents) - len(unique_docs)
        return unique_docs, unique_ids, removed_count
    
    def add_documents(
        self, 
        documents: List[Document], 
        custom_ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        写入文档（支持分批写入，避免单次数据量过大导致超时）
        :param documents: 文档列表
        :param custom_ids: 可选，自定义 ID。不传则自动生成
        :param batch_size: 批处理大小
        """
        if not documents:
            return []

        # 生成 ID
        if custom_ids is None:
            ids = [self._generate_unique_id(doc) for doc in documents]
        else:
            if len(custom_ids) != len(documents):
                raise ValueError("custom_ids 长度必须与 documents 一致")
            ids = custom_ids

        # 【修复点】在分批前先进行全局去重，避免同批次内 ID 冲突
        documents, ids, removed = self._deduplicate_by_id(documents, ids)
        if removed > 0:
            print(f"⚠️ 检测到 {removed} 条重复 ID，已自动去重，保留最新内容。")

        # 分批添加，提高稳定性
        all_added_ids = []
        total_batches = (len(documents) + batch_size - 1) // batch_size       
        # 进度条
        with tqdm(
            total=total_batches, 
            desc="Adding documents to Chroma", 
            unit="batch", 
            disable=not self.show_progress
        )as pbar:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                try:
                    self.db.add_documents(documents=batch_docs, ids=batch_ids)
                    all_added_ids.extend(batch_ids)
                except Exception as e:
                    # 如果遇到 ID 冲突等错误，可以选择跳过或报错
                    print(f"Batch {i//batch_size} failed: {e}")
                    raise e
        if self.show_progress:
            print(f"✅ 完成！共写入 {len(all_added_ids)} 条文档")
        return all_added_ids                

    def upsert_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """
        【推荐】幂等写入（更新或插入）
        逻辑：先计算 ID -> 删除已存在的 ID -> 插入新数据
        确保数据始终是最新的，且不会产生冲突报错
        """
        if not documents:
            return []
            
        # 生成 ID
        ids = [self._generate_unique_id(doc) for doc in documents]

        # 【修复点】批次内去重
        documents, ids, removed = self._deduplicate_by_id(documents, ids)
        if removed > 0:
            print(f"⚠️ Upsert 前检测到 {removed} 条重复 ID，已自动去重。")

        total_batches = (len(documents) + batch_size - 1) // batch_size
        all_added_ids = []

        with tqdm(
            total=total_batches * 2,  # 删除 + 插入 两步
            desc="🔄 更新向量库", 
            unit="batch",
            disable=not self.show_progress
        ) as pbar:   
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                # 删除旧数据
                if batch_ids:
                    self.db.delete(ids=batch_ids)
                    pbar.set_postfix({"阶段": "删除旧数据", "已处理": i + len(batch_docs)})
                    pbar.update(1)
                
                # 插入新数据
                self.db.add_documents(documents=batch_docs, ids=batch_ids)
                all_added_ids.extend(batch_ids)
                pbar.set_postfix({"阶段": "写入新数据", "已写入": len(all_added_ids)})
                pbar.update(1)

        if self.show_progress:
            print(f"✅ 完成！共更新 {len(all_added_ids)} 条文档")
        return all_added_ids

    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        相似度检索
        :param filter: 元数据过滤，例如 {"source": "guideline.pdf"}
        """
        return self.db.similarity_search(query=query, k=k, filter=filter)

    def get_document_by_ids(self, ids: List[str]) -> List[Document]:
        """
        根据 ID 获取文档内容 (用于验证或查看)
        """
        if not ids:
            return []
        # 获取底层集合
        collection = self.db._client.get_collection(self.collection_name)
        results = collection.get(ids=ids, include=["documents", "metadatas"])
        
        docs = []
        for i, content in enumerate(results['documents']):
            docs.append(Document(
                page_content=content,
                metadata=results['metadatas'][i] or {}
            ))
        return docs

    def delete_by_metadata_filter(self, filter: Dict[str, Any]) -> int:
        """
        根据元数据批量删除 (例如：删除某个来源的所有文档)
        返回删除的数量
        """
        try:
            # 先查询出符合条件的 ID
            collection = self.db._client.get_collection(self.collection_name)
            results = collection.get(where=filter, include=[])
            ids_to_delete = results['ids']
            
            if ids_to_delete:
                self.db.delete(ids=ids_to_delete)
                return len(ids_to_delete)
            return 0
        except Exception as e:
            print(f"Error deleting by filter: {e}")
            return 0

    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        根据唯一 ID 删除文档
        """
        if not ids:
            return False
        try:
            self.db.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息 (总数、唯一来源数等)
        """
        try:
            collection = self.db._client.get_collection(self.collection_name)
            total_count = collection.count()
            
            # 获取部分元数据分析来源分布
            # 注意：get 大量数据可能慢，这里仅获取 100 个样本做演示
            sample = collection.get(limit=100, include=["metadatas"])
            sources = set()
            for meta in sample['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
            
            return {
                "total_documents": total_count,
                "unique_sources_sample": len(sources),
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_collection(self):
        """
        清空当前集合 (慎用)
        """
        client = chromadb.PersistentClient(path=self.persist_directory)
        client.delete_collection(self.collection_name)
        # 重新初始化
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )