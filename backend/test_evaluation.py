# scripts/test_evaluation.py
import requests
import json
import time
from typing import List, Optional, Dict, Any

BASE_URL = "http://localhost:8000"

def test_get_questions():
    """测试获取问题列表"""
    print("\n📋 测试 1：获取问题列表")
    response = requests.get(f"{BASE_URL}/evaluation/questions")
    print(f"状态码：{response.status_code}")
    print(f"问题数量：{len(response.json())}")
    return response.json()

def test_single_evaluation(question_id="q001"):
    """测试单个问题评估"""
    print(f"\n🔍 测试 2：评估问题 {question_id}")
    response = requests.post(
        f"{BASE_URL}/evaluation/evaluate_single",
        params={"question_id": question_id}
    )
    print(f"状态码：{response.status_code}")
    result = response.json()
    print(f"评估结果：{json.dumps(result, ensure_ascii=False, indent=2)}")
    return result

def test_batch_evaluation():
    """测试批量评估"""
    print("\n🚀 测试 3：批量评估")
    response = requests.post(
        f"{BASE_URL}/evaluation/evaluate_batch",
        json={
            "dataset_name": "medical_benchmark_v1",
            "config": {"retrieval_mode": "hybrid_rerank", "top_k": 5}
        }
    )
    print(f"状态码：{response.status_code}")
    result = response.json()
    print(f"任务 ID: {result.get('data', {}).get('evaluation_id')}")
    return result

def test_get_reports():
    """测试获取报告列表"""
    print("\n📊 测试 4：获取报告列表")
    response = requests.get(f"{BASE_URL}/evaluation/reports")
    print(f"状态码：{response.status_code}")
    print(f"报告数量：{len(response.json())}")
    for report in response.json():
        print(f"  - {report.get('evaluation_id')}: {report.get('status')} ({report.get('avg_score', 0):.2f})")
    return response.json()

def test_add_question():
    """测试添加问题"""
    print("\n➕ 测试 5：添加测试问题")
    response = requests.post(
        f"{BASE_URL}/evaluation/questions",
        json={
            "id": f"q_test_{int(time.time())}",
            "query": "测试问题：高血压的早期症状有哪些？",
            "category": "diagnosis",
            "department": "cardiology",
            "difficulty": "easy"
        }
    )
    print(f"状态码：{response.status_code}")
    print(f"结果：{response.json()}")
    return response.json()

def test_submit_feedback():
    """测试提交反馈"""
    print("\n👥 测试 6：提交用户反馈")
    response = requests.post(
        f"{BASE_URL}/feedback/submit",
        json={
            "query": "测试反馈问题",
            "rating": 4,
            "comment": "回答比较准确，但可以增加更多引用",
            "issue_type": None
        }
    )
    print(f"状态码：{response.status_code}")
    print(f"结果：{response.json()}")
    return response.json()

def test_get_feedbacks():
    """测试获取反馈列表"""
    print("\n📝 测试 7：获取反馈列表")
    response = requests.get(f"{BASE_URL}/feedback/list")
    print(f"状态码：{response.status_code}")
    print(f"反馈数量：{len(response.json())}")
    return response.json()

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 医疗 RAG 测评功能测试")
    print("=" * 60)
    
    try:
        # 1. 获取问题列表
        questions = test_get_questions()
        
        if questions:
            # 2. 评估第一个问题
            test_single_evaluation(questions[0]["id"])
        
        # 3. 批量评估
        test_batch_evaluation()
        
        # 等待几秒让后台任务完成
        print("\n⏳ 等待 5 秒让评估任务完成...")
        time.sleep(5)
        
        # 4. 获取报告列表
        test_get_reports()
        
        # 5. 添加问题
        test_add_question()
        
        # 6. 提交反馈
        test_submit_feedback()
        
        # 7. 获取反馈列表
        test_get_feedbacks()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")

if __name__ == "__main__":
    run_all_tests()