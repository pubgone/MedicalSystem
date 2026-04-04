# 🏥 医疗知识 RAG 系统

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-blue.svg)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

**基于检索增强生成 (RAG) 的医疗知识问答系统**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [API 文档](#-api-文档) • [系统测评](#-系统测评) • [部署指南](#-部署指南)

</div>

---

## 📋 目录

- [项目简介](#-项目简介)
- [功能特性](#-功能特性)
- [快速开始](#-快速开始)

---

## 📖 项目简介

本项目是一个**企业级医疗知识 RAG（Retrieval-Augmented Generation）系统**，专为医疗场景设计，提供准确、安全、可追溯的医疗知识问答服务。

### 🎯 核心目标

- ✅ **准确性**：基于权威医疗文献，避免幻觉
- ✅ **安全性**：内置医疗安全合规检查
- ✅ **可追溯**：所有回答标注文献来源
- ✅ **可评估**：完整的测评体系监控质量

### 🏥 适用场景

| 场景 | 说明 |
|------|------|
| 医院知识库 | 医护人员快速查询诊疗指南 |
| 患者教育 | 提供权威健康科普信息 |
| 药学咨询 | 药品说明书查询与解读 |
| 医学研究 | 文献检索与知识整合 |

### ⚠️ 重要声明

> **本系统提供的信息仅供参考，不能替代专业医疗建议。所有医疗决策应在执业医师指导下进行。**

---

## ✨ 功能特性

### 🔍 智能检索

- **混合检索**：向量检索 + BM25 关键词检索
- **Rerank 重排序**：使用交叉编码器优化检索结果
- **元数据过滤**：支持科室、证据等级等多维度过滤
- **唯一标识**：文档去重与增量更新支持

### 💬 智能问答

- **多轮对话**：支持上下文关联的连续问答
- **流式输出**：实时显示生成过程，降低等待焦虑
- **引用标注**：每条信息标注来源文献
- **安全检测**：紧急情况自动识别并建议就医

### 📊 系统测评

- **自动化评估**：集成 RAGAS 评估框架
- **医疗专用指标**：安全性、合规性、幻觉检测
- **批量测试**：支持测试集批量评估
- **报告生成**：可视化评估报告

### 🛡️ 安全合规

- **免责声明**：所有回答自动添加医疗免责声明
- **紧急识别**：检测急症关键词并优先建议就医
- **剂量限制**：不提供具体处方药剂量建议
- **审计日志**：完整记录所有查询与回答

---

## 🚀 快速开始

### 第 0 步 下载代码安装依赖
```
#克隆仓库、安装依赖
git clone https://github.com/pubgone/MedicalSystem.git
cd backend && pip install -r requirements.txt
```

### 第 1 步 数据集下载及测试数据库建立

#### 📂 数据集放置说明
[医疗问答对](https://github.com/Toyhom/Chinese-medical-dialogue-data/tree/master)请下载数据集后，将所有文件放入项目的 `dataset/raw/` 目录下。

#### 创建测试向量数据库
创建测试向量数据库
```
python -m test.vectore_store_test
```

### 第 2 步 启动服务
```
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000  --reload
```

### 第 3 步 启动前端服务
前端源码存放于 frontend/ 目录，请自行构建并运行。



