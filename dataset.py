# # # #!/usr/bin/env python3
# # # # -*- coding: utf-8 -*-
# # # """
# # # LangChain + ChromaDB 最小化测试脚本
# # # 测试 Embedding + 向量库写入 + 检索
# # # """

# # # import os
# # # import sys
# # # from dotenv import load_dotenv

# # # # 加载环境变量（如果有 .env 文件）
# # # load_dotenv()



# # # import chromadb
# # # from langchain_community.embeddings import DashScopeEmbeddings

# # # print("1. 测试 ChromaDB...")
# # # client = chromadb.PersistentClient(path="./test_vec")
# # # collection = client.get_or_create_collection("test")
# # # print("✅ ChromaDB OK")

# # # print("2. 测试 Embedding...")
# # # embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key="sk-e4ef591b02444b37973055a090f0308d")
# # # vec = embeddings.embed_query("测试")
# # # print(f"✅ Embedding OK, 维度：{len(vec)}")

# # # print("3. 测试插入...")
# # # collection.add(ids=["test1"], documents=["测试文档"], embeddings=[vec], metadatas=[{"test": "ok"}])
# # # print(f"✅ 插入 OK, 总数：{collection.count()}")
# # # print("4. 测试插入...")
# # # collection.add(ids=["test2"], documents=["医学文档2"], embeddings=[vec], metadatas=[{"test": "ok"}])
# # # print(f"✅ 插入 OK, 总数：{collection.count()}")
# # # print("\n✅ 所有基础测试通过！")


# # import pandas as pd
# # import numpy as np
# # import os
# # import sys
# # import chromadb
# # from langchain_community.embeddings import DashScopeEmbeddings
# # from langchain_core.documents import Document
# # from langchain_community.vectorstores import Chroma
# # # ==================== 配置 ====================
# # CHROMA_PATH = "./medical_faq_vec"  # 向量库持久化路径
# # CSV_PATH = "dataset\样例_内科5000-6000.csv"      # 你的问答数据文件
# # COLLECTION_NAME = "medical_qa"     # 集合名称
# # DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-e4ef591b02444b37973055a090f0308d")
# # BATCH_SIZE = 100
    
# # # print(f"✅ 共处理 {len(docs)} 条问答数据")

# # # client = chromadb.PersistentClient(path=CHROMA_PATH)

# # # # 获取或创建 Collection
# # # collection = client.get_or_create_collection(
# # #     name=COLLECTION_NAME,
# # #     metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
# # # )

# # #配置Embedding
# # embeddings = DashScopeEmbeddings(
# #     model="text-embedding-v2",  
# #     dashscope_api_key=DASHSCOPE_API_KEY
# # )

# # #初始化Chroma持久化向量库
# # vector_store = Chroma(
# #     persist_directory=CHROMA_PATH,
# #     embedding_function=embeddings,
# #     collection_name=COLLECTION_NAME
# # )


# # # ==================== 4. 批量存入向量库 ====================
# # print(f"\n🚀 正在向量化并存入数据...")

# # df = pd.read_csv(CSV_PATH, encoding='gbk')
# # df = df.head(10)  # 🔥 只取前 10 条用于测试
# # # =============================================================
# # print(f"📊 测试数据量：{len(df)} 条")

# # total_inserted = 0
# # failed_batches = 0
# # BATCH_SIZE = 5  # 测试时可以用小批次，方便观察
# # docs = []

# # for i in range(0, len(df), BATCH_SIZE):
# #     batch_df = df.iloc[i:i+BATCH_SIZE]
# #     batch_num = i // BATCH_SIZE + 1
# #     total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
# #     try:
# #         # 准备 Document 列表
# #         docs = []
# #         for idx, row in batch_df.iterrows():
# #             content = f"问：{row['title']}\n答：{row['answer']}"
# #             metadata = {
# #                 "department": row['department'],
# #                 # "question": row['title'],
# #                 # "answer": row['answer'],  # 可选：answer 已在 content 中，metadata 尽量简洁
# #             }
# #             docs.append(Document(page_content=content, metadata=metadata))
# #         # 批量添加
# #         vector_store.add_documents(docs)
# #         # ✅ 修正：获取当前向量库总数
# #         current_count = vector_store._collection.count() if hasattr(vector_store, '_collection') else "N/A"
# #         total_inserted += len(docs)
# #         progress = (total_inserted / len(df)) * 100
# #         print(f"✅ 批次 {batch_num}/{total_batches} | "
# #               f"本批{len(docs)}条 | 累计{total_inserted}/{len(df)} ({progress:.1f}%) | "
# #               f"向量库总数：{current_count}")
              
# #     except Exception as e:
# #         print(f"⚠️ 批次 {batch_num} 失败：{e}")
# #         failed_batches += 1
# #         continue

# # # # ==================== 6. 检索测试 ====================
# # # print("\n【步骤 5】检索测试...")
# # # try:
# # #     # 创建检索器
# # #     retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
# # #     # 测试查询
# # #     test_queries = [
# # #         "高血压患者能吃党参吗？",
# # #         "糖尿病饮食注意事项",
# # #         "儿童发烧怎么办？"
# # #     ]
    
# # #     for query in test_queries:
# # #         print(f"\n❓ 查询：{query}")
# # #         results = retriever.invoke(query)
        
# # #         if results:
# # #             print(f"   ✅ 找到 {len(results)} 条结果")
# # #             for i, doc in enumerate(results[:2], 1):  # 只显示前 2 条
# # #                 print(f"\n   【结果{i}】")
# # #                 print(f"   科室：{doc.metadata.get('department', '未知')}")
# # #                 print(f"   问题：{doc.metadata.get('question', '未知')[:50]}...")
# # #                 print(f"   内容：{doc.page_content[:80]}...")
# # #         else:
# # #             print(f"   ⚠️ 未找到相关结果")
    
# # #     # 测试带过滤的检索
# # #     print(f"\n❓ 查询：发烧（过滤：儿科）")
# # #     results_filtered = vector_store.similarity_search(
# # #         query="发烧",
# # #         k=2,
# # #         filter={"department": "儿科"}
# # #     )
    
# # #     if results_filtered:
# # #         print(f"   ✅ 找到 {len(results_filtered)} 条结果")
# # #         for doc in results_filtered:
# # #             print(f"   科室：{doc.metadata.get('department', '未知')}")
# # #             print(f"   问题：{doc.metadata.get('question', '未知')[:50]}...")
# # #     else:
# # #         print(f"   ⚠️ 未找到相关结果")
    
# # # except Exception as e:
# # #     print(f"⚠️ 检索测试失败：{e}")
# # #     print("💡 不影响向量库使用，可手动测试")

# # # # ==================== 7. 封装检索函数（供后续使用）====================
# # # print("\n" + "="*60)
# # # print("💡 后续使用示例代码")
# # # print("="*60)

# # # print("""
# # # # 加载已存在的向量库
# # # from langchain_community.vectorstores import Chroma
# # # from langchain_community.embeddings import DashScopeEmbeddings

# # # embeddings = DashScopeEmbeddings(
# # #     model="text-embedding-v2",
# # #     dashscope_api_key="your-api-key"
# # # )

# # # vector_store = Chroma(
# # #     persist_directory="./medical_faq_vec",
# # #     embedding_function=embeddings,
# # #     collection_name="medical_qa"
# # # )

# # # # 检索
# # # retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# # # results = retriever.invoke("高血压怎么治疗？")

# # # # 带科室过滤
# # # results = vector_store.similarity_search(
# # #     query="发烧",
# # #     k=3,
# # #     filter={"department": "儿科"}
# # # )
# # # """)

# # # print("="*60)
# # # print("✅ 所有步骤完成！")
# # # print("="*60)



# # # #存入向量库
# # # vector_store = Chroma(
# # #     client=client,
# # #     collection_name="medical_qa",
# # #     embedding_function=embeddings
# # # )
# # # print(f"✅ 向量库构建完成，共 {vector_store._collection.count()} 条向量")



# # # model = BGEM3FlagModel('BAAI/bge-m3',  
# # #                        use_fp16=True) 
# # # #向量化
# # # # 2. 准备数据
# # # texts = [doc.page_content for doc in docs]
# # # metadatas = [doc.metadata for doc in docs]

# # # #计算向量
# # # output = model.encode(texts, batch_size=32, max_length=512)
# # # vectors = output['dense_vecs']  # 只取稠密向量用于 Chroma/FAISS

# # # # 3. 创建向量数据库
# # # embeddings = BGE3Embeddings(model_name='BAAI/bge-m3', use_fp16=True)
# # # vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./medical_faq_vec")

# # # # 4. 检索使用（支持科室过滤）
# # # retriever = vector_store.as_retriever(search_kwargs={"k": 3, "filter": {"department": "心血管内科"}})
# # # results = retriever.invoke("高血压患者能吃党参吗？")
import pandas as pd
import numpy as np
import os
import sys
import chromadb
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# 4. 空值处理函数
def clean_metadata_value(val):
    """
    清理元数据值，处理空值、NaN、None 等情况
    返回干净的字符串，如果为空则返回 'unknown'
    """
    if val is None:
        return 'unknown'
    if isinstance(val, float) and pd.isna(val):
        return 'unknown'
    val_str = str(val).strip()
    if val_str == '' or val_str.lower() == 'nan' or val_str.lower() == 'none':
        return 'unknown'
    return val_str

# ==================== 配置 ====================
CHROMA_PATH = "./medical_faq_vec"  # 向量库持久化路径
CSV_PATH = "dataset\样例_内科5000-6000.csv"      # 你的问答数据文件
COLLECTION_NAME = "medical_qa"     # 集合名称
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-e4ef591b02444b37973055a090f0308d")
BATCH_SIZE = 100
CONTENT_COLUMN = "answer"  # 你想用来生成向量的列名，替换为实际列名
METADATA_COLUMNS = ["department", "question"]  # 作为元数据的列名 # 你想保留为元数据的列名列表，替换为实际列名
##############################################################################
#####################################配置Embedding#############################
###############################################################################
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  
    dashscope_api_key=DASHSCOPE_API_KEY
)

##############################################################################
#####################################导入数据##################################
###############################################################################
print(f"\n🚀 正在存入数据...")

df = pd.read_csv(CSV_PATH, encoding='gbk', nrows=10)
docs=[]
metadata = {}
for idx,row in df.iterrows():
    content = clean_metadata_value(row.get('answer', ''))
    # 🔹 metadata: 放入 department 和 question
    metadata = {
        'department': clean_metadata_value(row.get('department', '')),
        'title': clean_metadata_value(row.get('title', ''))
    }
    doc = Document(page_content=content, metadata=metadata)
    docs.append(doc)
        # 打印预览
    print(f"\n[{idx+1}] Document 预览:")
    print(f"  📝 content: {content[:60]}...")
    print(f"  🏷️  metadata: {metadata}")
print(f"\n📄 Document 构建完成:")
print(f"  内容长度：{len(doc.page_content)} 字符")
print(f"  元数据：{doc.metadata}")
vector_store = Chroma(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)
print(f"✅ 成功写入 1 条文档到 Chroma")
# # =============================================================
# print(f"📊 测试数据量：{len(df)} 条")

# total_inserted = 0
# failed_batches = 0
# BATCH_SIZE = 5  # 测试时可以用小批次，方便观察
# docs = []

# for i in range(0, len(df), BATCH_SIZE):
#     batch_df = df.iloc[i:i+BATCH_SIZE]
#     batch_num = i // BATCH_SIZE + 1
#     total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

#     docs = []
#     for idx, row in batch_df.iterrows():
#         content = f"问：{row['title']}\n答：{row['answer']}"
#         metadata = {
#             "department": row['department'],
#             "question": row['title']
#             # "answer": row['answer'],  # 可选：answer 已在 content 中，metadata 尽量简洁
#         }
#         docs.append(Document(page_content=content, metadata=metadata))
#     print(f"type(docs): {type(docs[0])}, len(docs): {len(docs)}")
#     print(docs[4])
# vectorstore = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     persist_directory="./chroma_db",
# )

# print(2)
# total_inserted += len(docs)
# progress = (total_inserted / len(df)) * 100
# print(f"✅ 批次 {batch_num}/{total_batches} | "
#       f"已插入 {total_inserted}/{len(df)} ({progress:.1f}%)")
          
# import os
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.documents import Document

# # 1. 设置 API Key (建议放在环境变量中)
# os.environ["DASHSCOPE_API_KEY"] = "sk-e4ef591b02444b37973055a090f0308d"  # 替换为你的真实 Key
# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# # 2. 初始化 Embeddings (你提供的代码)
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v2",  
#     dashscope_api_key=DASHSCOPE_API_KEY
# )

# # 3. 准备一些测试文本
# texts = [
#     "阿里巴巴通义千问是大语言模型",
#     "Chroma 是一个开源的向量数据库",
#     "LangChain 用于构建 LLM 应用"
# ]
# documents = [Document(page_content=text) for text in texts]

# # 4. 初始化 Chroma 并传入 embeddings
# # persist_directory 是本地存储路径，如果不填则只在内存中
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=embeddings,  # <--- 这里使用了你创建的 embeddings 对象
#     persist_directory="./chroma_db"
# )

# # 5. 测试检索
# query = "什么是通义千问？"
# results = vectorstore.similarity_search(query, k=1)

# print("检索结果:", results[0].page_content)