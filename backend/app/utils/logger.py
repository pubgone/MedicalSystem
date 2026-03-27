# backend/app/utils/logger.py
import logging
import sys
from pathlib import Path
from typing import Optional

# 全局 logger 实例
_logger: Optional[logging.Logger] = None

def setup_logger(
    name: str = "medical_rag",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回 logger 实例
    :param name: logger 名称
    :param level: 日志级别
    :param log_file: 日志文件路径（可选）
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler
    if logger.handlers:
        _logger = logger
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger

def get_logger(name: str = "medical_rag") -> logging.Logger:
    """
    获取 logger 实例
    :param name: logger 名称
    :return: logging.Logger 实例
    """
    global _logger
    if _logger is None:
        # 自动初始化
        return setup_logger(name)
    return _logger

# 模块加载时自动初始化
setup_logger()