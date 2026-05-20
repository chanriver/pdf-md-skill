---
name: pdf-to-md
description: |
  通用 PDF 转 Markdown 工作流，支持 PDF 提取和 LLM 优化排版。
  当用户提到以下场景时触发：
  - PDF 转 MD / Markdown
  - PDF 转文档并优化排版
  - 用大模型优化 PDF 提取的 MD
  - 文档格式转换（PDF → MD）
  此 SKILL 支持 mineru API 提取 + 多 Provider LLM 优化，可用于 Claude Code 及各类 AI Agent 平台。
---

# PDF to MD 通用工作流

## 概述

通用 PDF 转 Markdown 工作流，支持：
1. **PDF 提取** - 使用 Mineru API 将 PDF 转为 MD
2. **LLM 优化** - 使用大模型优化排版（支持多 Provider）
3. **图床上传** - 可选上传图片到图床

兼容 Claude Code 及各类 AI Agent 平台（CrewAI、AutoGen、LangChain 等）。

## 工作流程

```
PDF 文件 → Mineru 提取 → MD 文件 → LLM 优化 → 优化后 MD
```

## 使用方法

### Claude Code 中使用

```
用户: 帮我把这个 PDF 转成 MD
→ 调用本 Skill，自动执行完整流程
```

### Hermes Agent / OpenClaw 中使用

Hermes Agent / OpenClaw 使用 MiniMax M2.7 大模型，配置方式：

```python
from pdf_md_workflow import PDFMDWorkflow
from openai import OpenAI

# Hermes Agent / OpenClaw 的 MiniMax 配置
client = OpenAI(
    api_key="your-minimax-api-key",
    base_url="https://api.minimaxi.com/v1"
)

workflow = PDFMDWorkflow(
    llm_client=client,
    llm_model="MiniMax-M2.7",
    llm_provider_type="minimax"
)

result = workflow.process("document.pdf")
# 返回: {"md_path": "...", "optimized_path": "...", ...}
```

### 其他 Agent 平台调用

```python
from pdf_md_workflow import PDFMDWorkflow

# 方式一：使用已有的 LLM Client（推荐）
# Agent 平台已配置大模型，无需重复配置
import anthropic
client = anthropic.Anthropic()  # 使用平台已有配置

workflow = PDFMDWorkflow(
    llm_client=client,                    # 传入已有的 Client
    llm_model="claude-opus-4-6",
    llm_provider_type="anthropic"         # "anthropic" | "openai" | "minimax"
)

result = workflow.process("document.pdf")

# 方式二：不传 Client，从配置/环境变量读取
workflow = PDFMDWorkflow(config_path="config.yaml")
result = workflow.process("document.pdf")
```

## 配置说明

### API Keys（优先级：环境变量 > config.yaml）

| 服务 | 环境变量 | 说明 |
|------|----------|------|
| Mineru | MINERU_API_KEY | PDF 提取服务 |
| Anthropic | ANTHROPIC_API_KEY | Claude 模型 |
| OpenAI | OPENAI_API_KEY | GPT 模型 |
| MiniMax | MINIMAX_API_KEY | MiniMax 模型（Hermes/OpenClaw 默认） |

### LLM Provider 配置

```yaml
default_provider: "minimax"

anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-opus-4-6"

openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o"

minimax:
  api_key: "${MINIMAX_API_KEY}"
  base_url: "https://api.minimaxi.com/v1"
  model: "MiniMax-M2.7"

minimax:
  api_key: "${MINIMAX_API_KEY}"
  base_url: "https://api.minimaxi.com/v1"
  model: "MiniMax-M2.7"
```

## 通用优化规则

LLM 优化包含以下通用规则（可自定义）：

### 1. 标题层级修复
- 正确识别章节层级（如 1 → ##, 1.1 → ###）
- 去除错误的页码后缀

### 2. 公式处理
- 公式块 `$$...$$` 原样保留
- 行内公式智能处理

### 3. 表格转换
- 简单表格：HTML → Markdown
- 复杂表格（rowspan/colspan）：保留 HTML

### 4. 代码块处理
- 保留原始代码内容
- 正确识别语言标识符

### 5. 格式清理
- 去除多余空行
- 修复常见乱码
- 统一空格使用

## CLI 用法

```bash
# 完整流程（PDF → 提取 → 优化）
python scripts/pdf_md_workflow.py --pdf document.pdf

# 只提取 PDF
python scripts/pdf_md_workflow.py --extract document.pdf

# 只优化已有 MD
python scripts/pdf_md_workflow.py --optimize document.md

# 批量处理
python scripts/pdf_md_workflow.py --batch "*.pdf"

# 指定 Provider
python scripts/pdf_md_workflow.py --pdf doc.pdf --provider openai

# 跳过图床上传
python scripts/pdf_md_workflow.py --pdf doc.pdf --no-image
```

## 依赖安装

```bash
pip install anthropic openai pyyaml requests
```

## 文件结构

```
pdf-to-md通用版/
├── SKILL.md                    # Skill 元数据（本文件）
├── README.md                   # 详细文档（可选）
└── scripts/
    ├── pdf_md_workflow.py      # 主工作流脚本（包含所有 Provider）
    ├── config.yaml             # 用户配置（可选）
    └── config.yaml.example     # 配置模板
```

> 注意：所有 LLM Provider（Anthropic、OpenAI、MiniMax、NVIDIA）已集成在 `pdf_md_workflow.py` 中，无需单独模块。
