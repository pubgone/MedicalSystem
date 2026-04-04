from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from rag.vector_store import VectorStore
from utils.data_process import MedicalKnowledgeLoader
from config.config import config

embedding_model = config.EMBEDDING_MODEL
vector_store = VectorStore(
    collection_name="medical_knowledge_v1",
    persist_directory="./chroma_db",
    show_progress=True
)


# --- 场景 1：当前状态 (只有 source) ---
loader = MedicalKnowledgeLoader(
        content_column='answer',
        metadata_columns=['ask', 'department']
    )
docs = loader.load(r"dataset\raw\csv")

# --- 场景 2：未来升级 (添加了 page) ---
# 假设后续数据处理流程优化，提取出了页码
# docs_future = [
#     Document(
#         page_content="高血压患者应每日监测血压。", # 内容一样
#         metadata={"source": "高血压指南_2023.pdf", "page": 12, "department": "cardiology"}
#         # 注意：这里多了 page 字段
#     )
# ]
# ids_2 = vector_store.upsert_documents(docs_future)
ids_1 = vector_store.upsert_documents(docs)