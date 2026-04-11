# 使用本地打包的 conda 环境
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 复制打包的环境
COPY langchain-env.tar.gz /tmp/langchain-env.tar.gz

# 安装 Miniconda (使用清华镜像，禁用SSL验证)
RUN curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda config --set ssl_verify false

# 配置 conda
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=langchain

# 解压环境
RUN /opt/conda/bin/conda-unpack -p /opt/conda -s /tmp/langchain-env.tar.gz 2>/dev/null || \
    (mkdir -p /opt/conda && tar -xzf /tmp/langchain-env.tar.gz -C /opt/conda --strip-components=1)

# 复制应用代码
COPY backend/ /app/

# 创建向量数据库目录
RUN mkdir -p /app/chroma_db

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/live || exit 1

# 启动命令
CMD ["/opt/conda/envs/langchain/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
