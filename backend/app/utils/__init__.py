# backend/app/utils/logger.py
import logging
import sys
from pathlib import Path

def setup_logger():
    """配置日志"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    # 创建 logger
    logger = logging.getLogger("medical_rag")
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "medical_rag"):
    return logging.getLogger(name)

# 初始化
setup_logger()