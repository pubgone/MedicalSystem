# backend/app/evaluation/ragas_evaluator.py
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from backend.app.api.schemas import (
    EvaluateSingleRequest, EvaluateBatchRequest, CompareModesRequest,
    EvaluationReport, EvaluationResponse, CompareModesResponse,
    APIResponse
)


class MedicalRAGASEvaluator:
    """医疗场景 RAGAS 评估器"""
    
    def __init__(self, llm, embeddings, run_config: RunConfig = None):
        """
        :param llm: LangChain LLM 实例
        :param embeddings: LangChain Embeddings 实例
        :param run_config: RAGAS 运行配置（并发/超时等）
        """
        self.llm_wrapper = LangchainLLMWrapper(llm)
        self.embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)
        self.run_config = run_config or RunConfig(timeout=180, max_workers=4)
        
        # 医疗场景核心指标（按重要性排序）
        self.metrics = [
            faithfulness,           # ⭐ 防幻觉（最重要）
            answer_relevancy,       # 答案相关性
            context_precision,      # 检索精准度
            context_recall,         # 检索召回率
            answer_correctness,     # 答案正确性（需 ground truth）
        ]
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
        save_path: str = None
    ) -> Dict[str, Any]:
        """
        批量评估
        :param questions: 问题列表
        :param answers: RAG 生成的答案
        :param contexts: 每个问题对应的检索上下文（列表的列表）
        :param ground_truths: 标准答案（可选，用于 answer_correctness）
        """
        # 构建评估数据集
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # 执行评估
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm_wrapper,
            embeddings=self.embeddings_wrapper,
            run_config=self.run_config,
            raise_exceptions=False  # 避免单个失败中断整体
        )
        
        # 转换为字典
        scores = result.to_pandas().to_dict(orient='records')
        aggregate = result.to_pandas().mean().to_dict()
        
        # 保存详细结果
        if save_path:
            df = result.to_pandas()
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(questions),
            "aggregate_scores": {k: round(v, 4) for k, v in aggregate.items()},
            "individual_scores": scores,
            "recommendations": self._generate_recommendations(aggregate)
        }
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """根据得分生成优化建议"""
        recommendations = []
        
        if scores.get("faithfulness", 0) < 0.7:
            recommendations.append("⚠️ Faithfulness 偏低：答案可能包含幻觉，建议：\n"
                                 "  - 增加 Rerank 步骤提升上下文质量\n"
                                 "  - 在 Prompt 中强化'仅基于上下文回答'指令")
        
        if scores.get("context_recall", 0) < 0.6:
            recommendations.append("⚠️ Context Recall 偏低：相关文档可能未被检索到，建议：\n"
                                 "  - 扩大 top_k 或使用混合检索\n"
                                 "  - 检查 embedding 模型是否适合医疗领域")
        
        if scores.get("answer_relevancy", 0) < 0.75:
            recommendations.append("⚠️ Answer Relevance 偏低：答案可能偏离问题，建议：\n"
                                 "  - 优化 Prompt 结构，明确要求'直接回答问题'\n"
                                 "  - 添加问题重述步骤确保理解正确")
        
        if scores.get("context_precision", 0) < 0.7:
            recommendations.append("⚠️ Context Precision 偏低：检索结果包含较多噪声，建议：\n"
                                 "  - 添加 BM25 + 向量混合检索\n"
                                 "  - 使用 Rerank 模型重排序")
        
        if not recommendations:
            recommendations.append("✅ 各项指标良好，可考虑：\n"
                                 "  - 扩大测试集覆盖更多医疗场景\n"
                                 "  - 添加线上 A/B 测试验证用户满意度")
        
        return recommendations
    
    def compare_configs(
        self,
        test_cases: List[Dict[str, Any]],
        configs: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        对比不同配置的效果（A/B 测试）
        :param test_cases: 测试用例列表，每项包含 {question, ground_truth, relevant_docs}
        :param configs: 配置列表，每项包含 {name, retrieval_mode, top_k, prompt_template}
        """
        results = []
        
        for config in configs:
            print(f"🔍 评估配置：{config['name']}")
            
            questions = [tc["question"] for tc in test_cases]
            ground_truths = [tc["ground_truth"] for tc in test_cases]
            
            # 使用不同配置生成答案和上下文（需要传入 rag_service）
            # 这里简化为占位，实际需调用你的 RAG 服务
            answers = []  # = [rag_service.chat(q, **config)["answer"] for q in questions]
            contexts = []  # = [rag_service.chat(q, **config)["documents"] for q in questions]
            
            result = self.evaluate_batch(questions, answers, contexts, ground_truths)
            
            results.append({
                "config_name": config["name"],
                **result["aggregate_scores"]
            })
        
        return pd.DataFrame(results).sort_values("faithfulness", ascending=False)