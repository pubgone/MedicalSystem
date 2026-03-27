from langchain_qwq import ChatQwen
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
from config.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        logger.info("初始化RAG引擎")

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name = config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu", 'local_files_only': True },
            encode_kwargs = {'normalize_embeddings': True}
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_langchain_db",
            embedding_function=self.embeddings,
            collection_name="example_collection"
        )
        
        
        self.llm = ChatQwen(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-e4ef591b02444b37973055a090f0308d",
            model="qwen-flash",
            max_tokens=3_000,
            timeout=None,
            max_retries=2
        )
                # 4. 构建 RAG 链
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )
        self.prompt = ChatPromptTemplate.from_template("""
你是一个医疗知识问答助手。请根据以上上下文回答问题。

上下文信息:
{context}
用户问题:
{question}                                                       
要求:
1. 仅根据上下文回答，不要编造信息
2. 如果上下文中没有答案，请明确说明
3. 回答要简洁、准确、专业
4. 如有必要，注明信息来源
                                            
回答:
"""
        )
        self.rag_chain = (
            {"context": self.retriever, "question":RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        # 5. 流式链（用于 SSE 输出）
        self.llm_stream = ChatQwen(
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL,
            model=config.QWEN_MODEL,
            streaming=False
        )
        self.rag_chain_stream = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm_stream
            | StrOutputParser()
        )
        
        logger.info("RAG 引擎初始化完成")
    def query(self, question: str,  top_k: int = 3)->Dict[str, Any]:
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k":top_k})
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            docs = retriever.invoke(question) 
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", None)
                }
                for doc in docs
            ]
            # 生成回答
            answer = chain.invoke(question)
            
            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }
        except Exception as e:
            logger.error(f"查询失败：{str(e)}")
            raise

    def query_stream(self, question: str, top_k: int = 3):
        """流式查询（生成器）"""
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm_stream
                | StrOutputParser()
            )
            
            for chunk in chain.stream(question):
                yield chunk
        except Exception as e:
            logger.error(f"流式查询失败：{str(e)}")
            raise

# 全局单例
rag_engine = RAGEngine()            