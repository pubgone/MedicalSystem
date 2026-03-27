# backend/app/test_config.py
import sys
from pathlib import Path

# 确保可以导入 app 模块
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.config import settings
    
    print("🔍 配置加载测试")
    print("=" * 40)
    print(f"APP_NAME: {settings.APP_NAME}")
    print(f"LLM_MODEL: {settings.LLM_NAME}")
    print(f"DASHSCOPE_API_KEY: {'✓ 已配置' if settings.LLM_KEY else '✗ 未配置'}")
    print(f"API_KEY: {'✓ 已配置' if settings.API_KEY else '✗ 未配置'}")
    print(f"DEVICE: {settings.DEVICE}")
    print(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    print("=" * 40)
    
    if not settings.LLM_KEY:
        print("⚠️  警告：DASHSCOPE_API_KEY 为空！")
        print("   请检查 .env 文件或 docker-compose environment 配置")
    else:
        print("✅ 配置加载成功！")
        
except ImportError as e:
    print(f"❌ 导入错误：{e}")
    print(f"   当前路径：{Path.cwd()}")
    print(f"   sys.path: {sys.path}")
except Exception as e:
    print(f"❌ 错误：{e}")