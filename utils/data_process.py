import os
import csv
from pathlib import Path
from typing import List, Dict, Optional, Literal
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

class MedicalKnowledgeLoader:
    """
    医疗知识数据加载器 - 支持文件/文件夹自动识别
    """
    
    # 支持的文件扩展名配置
    SUPPORTED_EXTENSIONS = {'.csv', '.txt', '.pdf', '.md'}
    
    def __init__(self, content_column: str = 'content', metadata_columns: Optional[List[str]] = None):
        self.content_column = content_column
        self.metadata_columns = metadata_columns or []
    
    def load(self, path: str, 
             chunk_strategy: Literal['none', 'character', 'semantic'] = 'none',
             chunk_size: int = 500,
             chunk_overlap: int = 50,
             semantic_threshold: float = 0.5,       
             recursive: bool = True, 
             **kwargs) -> List[Document]:
        """
        智能加载：自动识别文件路径或文件夹路径
        
        Args:
            path: 文件或文件夹路径
            chunk_strategy: 分块策略
                - 'none': 不分块 (适合 CSV 行级数据)
                - 'character': 字符递归分块 (通用)
                - 'semantic': 语义分块 (效果最好，推荐医疗场景)
            chunk_size: 字符分块大小
            chunk_overlap: 字符分块重叠
            semantic_threshold: 语义分块阈值 (0-1)，越低分块越细
            recursive: 是否递归遍历文件夹
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"路径不存在：{path}")
        
        # 1. 收集原始文档
        raw_docs = []
        if path_obj.is_file():
            raw_docs = self._process_single_file(path_obj, **kwargs)
        elif path_obj.is_dir():
            raw_docs = self._process_directory(path_obj, recursive, **kwargs)
        
        if not raw_docs:
            return []
        
        # 2. 应用分块策略
        if chunk_strategy == 'character':
            return self._split_by_character(raw_docs, chunk_size, chunk_overlap)
        elif chunk_strategy == 'semantic':
            return self._split_by_semantic(raw_docs, semantic_threshold, chunk_size)
        else:
            return raw_docs
    def _split_by_character(self, docs: List[Document], size: int, overlap: int) -> List[Document]:
        """字符递归分块"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        return splitter.split_documents(docs)
    def _split_by_semantic(self, docs: List[Document], threshold: float, max_chunk_size: int) -> List[Document]:
        """
        语义分块：基于句子嵌入相似度
        当相邻句子相似度 < threshold 时，在此处切分
        """
        if not SEMANTIC_AVAILABLE:
            print("[Warn] 未安装 sentence-transformers，降级为字符分块")
            return self._split_by_character(docs, max_chunk_size, max_chunk_size // 10)
        
        # 懒加载嵌入模型 (使用中文医疗友好的模型)
        if self._embedding_model is None:
            print("[Info] 加载语义分块模型 (bge-small-zh-v1.5)...")
            self._embedding_model = SentenceTransformer('bge-small-zh-v1.5')
        
        chunked_docs = []
        
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # 短文本不需要分块
            if len(content) < max_chunk_size:
                chunked_docs.append(doc)
                continue
            
            # 1. 按句子分割
            sentences = self._split_into_sentences(content)
            if len(sentences) <= 1:
                chunked_docs.append(doc)
                continue
            
            # 2. 计算句子嵌入
            embeddings = self._embedding_model.encode(sentences, show_progress_bar=False)
            
            # 3. 计算相邻句子相似度
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(sim)
            
            # 4. 根据相似度切分
            chunks = []
            current_chunk = [sentences[0]]
            
            for i, sim in enumerate(similarities):
                # 相似度低于阈值，说明语义发生变化，在此切分
                if sim < threshold:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                current_chunk.append(sentences[i + 1])
            
            if current_chunk:
                chunks.append("".join(current_chunk))
            
            # 5. 创建文档对象
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    chunk_meta = metadata.copy()
                    chunk_meta["chunk_index"] = i
                    chunk_meta["total_chunks"] = len(chunks)
                    chunk_meta["chunk_type"] = "semantic"
                    chunked_docs.append(Document(page_content=chunk, metadata=chunk_meta))
        
        print(f"[Semantic] 分块完成：{len(docs)} -> {len(chunked_docs)} 个语义片段")
        return chunked_docs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """按中文标点分割句子"""
        import re
        # 保留标点符号
        sentences = re.split(r'(?<=[。！？；\n])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """计算余弦相似度"""
        import numpy as np
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


    def _process_single_file(self, file_path: Path, **kwargs) -> List[Document]:
        """处理单个文件"""
        ext = file_path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            print(f"[Skip] 不支持的文件类型：{file_path.name}")
            return []
        
        if ext == '.csv':
            return self._load_csv(str(file_path), **kwargs)
        elif ext == '.txt':
            return self._load_txt(str(file_path), **kwargs)
        elif ext == '.pdf':
            return self._load_pdf(str(file_path), **kwargs)
        elif ext == '.md':
            return self._load_md(str(file_path), **kwargs)
        
        return []
    
    def _process_directory(self, dir_path: Path, recursive: bool, **kwargs) -> List[Document]:
        """处理文件夹"""
        all_documents = []
        file_count = 0
        error_files = []
        
        # 选择遍历方式：递归 (rglob) 或 单层 (glob)
        pattern = "**/*" if recursive else "*"
        
        print(f"[Scan] 开始扫描目录：{dir_path} (递归：{recursive})")
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                try:
                    docs = self._process_single_file(file_path, **kwargs)
                    all_documents.extend(docs)
                    if docs:
                        file_count += 1
                except Exception as e:
                    error_files.append((file_path.name, str(e)))
                    print(f"[Error] 处理文件失败 {file_path.name}: {e}")
        
        # 打印处理摘要
        print(f"[Done] 扫描完成。成功处理 {file_count} 个文件，共 {len(all_documents)} 个文档片段。")
        if error_files:
            print(f"[Warn] {len(error_files)} 个文件处理失败：{[f[0] for f in error_files]}")
            
        return all_documents
    
    def _load_csv(self, file_path: str, encoding: str = 'utf-8', **kwargs) -> List[Document]:
        """加载 CSV 文件（支持自动编码检测）"""
        documents = []
        # 尝试多种编码
        for enc in [encoding, 'gbk', 'utf-8-sig']:
            try:
                with open(file_path, 'r', encoding=enc, newline='') as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames or self.content_column not in reader.fieldnames:
                        # 如果列名不匹配，不尝试其他编码，直接报错或跳过
                        if enc == encoding: 
                            raise ValueError(f"未找到内容列 '{self.content_column}'，可用列：{reader.fieldnames}")
                        continue 
                    
                    for row_num, row in enumerate(reader, start=2):
                        content = row.get(self.content_column, "").strip()
                        if not content:
                            continue
                        
                        metadata = {
                            "source": os.path.basename(file_path),
                            "row_index": row_num,
                            "full_path": file_path
                        }
                        for col in self.metadata_columns:
                            if col in row:
                                metadata[col] = row[col]
                        
                        documents.append(Document(page_content=content, metadata=metadata))
                break  # 成功读取后跳出编码尝试循环
            except UnicodeDecodeError:
                continue
        
        return documents
    
    def _load_txt(self, file_path: str, **kwargs) -> List[Document]:
        """加载 TXT 文件 - 预留接口"""
        # TODO: 实现 TXT 解析（例如按空行分割段落）
        raise NotImplementedError("TXT 文件加载尚未实现")
    
    def _load_pdf(self, file_path: str, **kwargs) -> List[Document]:
        """加载 PDF 文件 - 预留接口"""
        # TODO: 实现 PDF 解析
        raise NotImplementedError("PDF 文件加载尚未实现")

    def _load_md(self, file_path: str, **kwargs) -> List[Document]:
        """加载 Markdown 文件 - 预留接口"""
        # TODO: 实现 MD 解析
        raise NotImplementedError("Markdown 文件加载尚未实现")

