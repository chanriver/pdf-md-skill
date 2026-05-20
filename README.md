# PDF to MD Universal Workflow

通用 PDF 转 Markdown 工作流，支持多 Provider LLM 优化排版。可用于 Claude Code、Hermes Agent、OpenClaw 及各类 AI Agent 平台。

## 功能特性

- **PDF 提取** - 使用 Mineru API 将 PDF 转为 Markdown
- **LLM 优化** - 使用大模型智能优化排版（支持多 Provider）
- **智能图片处理**
  - 表格图片 → 用 LLM 转换为 Markdown 表格
  - 普通图片 → 上传到图床，替换链接，删除本地文件
- **多平台兼容** - Claude Code、Hermes Agent、OpenClaw、CrewAI、AutoGen、LangChain

## 支持的 LLM Provider

| Provider | 模型 | 说明 |
|----------|------|------|
| MiniMax | MiniMax-M2.7 | Hermes Agent / OpenClaw 默认 |
| Anthropic | Claude Opus/Sonnet/Haiku | Claude 模型 |
| OpenAI | GPT-4o / GPT-4 | OpenAI 模型 |
| NVIDIA | DeepSeek-V3.2 | 备用选项 |

## 目录结构

```
pdf-to-md通用版/
├── README.md               # 本文档
├── SKILL.md                # Claude Code Skill 元数据
└── scripts/
    ├── pdf_md_workflow.py  # 主工作流脚本
    └── config.yaml.example # 配置模板
```

## 快速开始

### 1. 安装依赖

```bash
pip install anthropic openai pyyaml requests
```

### 2. 配置环境变量

```bash
# Mineru API (PDF 提取)
export MINERU_API_KEY="your-mineru-api-key"

# LLM Provider (四选一，根据你的 Agent 平台选择)
export MINIMAX_API_KEY="your-minimax-api-key"
# 或
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# 或
export OPENAI_API_KEY="your-openai-api-key"
# 或
export NVIDIA_API_KEY="your-nvidia-api-key"

# 图床 Token (可选)
export IMAGE_HOST_TOKEN="your-image-token"
```

### 3. 使用

#### Claude Code

```
用户: 帮我把这个 PDF 转成 MD
→ 调用本 Skill，自动执行完整流程
```

#### Python 脚本

```bash
# 完整流程（PDF → 提取 → 优化）
python scripts/pdf_md_workflow.py --pdf document.pdf

# 仅提取 PDF
python scripts/pdf_md_workflow.py --extract document.pdf

# 仅优化已有 MD
python scripts/pdf_md_workflow.py --optimize document.md

# 批量处理
python scripts/pdf_md_workflow.py --batch "*.pdf"

# 跳过图片处理
python scripts/pdf_md_workflow.py --pdf doc.pdf --no-image
```

## 图片处理流程

PDF 提取的图片分为两类处理：

### 1. 表格图片
- 用 LLM 视觉能力识别图片中的表格数据
- 转换为 Markdown 表格格式
- 直接替换 MD 中的图片引用

### 2. 普通图片
- 上传到图床（`imgbed.361026.xyz`）
- 获取图床链接
- 替换 MD 中的图片引用为图床链接
- **删除本地图片文件**

### 判断逻辑

```python
# LLM 判断图片类型
if 图片是表格 → 调用 LLM 转换为 Markdown 表格
else → 上传到图床，替换链接，删除本地文件
```

> 注意：图片处理需要配置图床 Token 和 LLM Client。

## Hermes Agent / OpenClaw 配置

Hermes Agent 和 OpenClaw 使用 MiniMax M2.7 大模型：

```python
from pdf_md_workflow import PDFMDWorkflow
from openai import OpenAI

# 创建 MiniMax Client
client = OpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url="https://api.minimaxi.com/v1"
)

workflow = PDFMDWorkflow(
    llm_client=client,
    llm_model="MiniMax-M2.7",
    llm_provider_type="minimax"
)

result = workflow.process("document.pdf")
```

## 其他 Agent 平台集成

### CrewAI

```python
from crewai import Agent, Task, Tool
import sys
sys.path.insert(0, '/path/to/pdf-to-md通用版/scripts')
from pdf_md_workflow import PDFMDWorkflow

workflow = PDFMDWorkflow(
    llm_client=crewai_llm_client,
    llm_model="claude-opus-4-6",
    llm_provider_type="anthropic"
)

pdf_tool = Tool(
    name="PDF转MD",
    func=lambda x: workflow.process(x),
    description="将PDF文件转换为优化后的Markdown"
)
```

### AutoGen

```python
import sys
sys.path.insert(0, '/path/to/pdf-to-md通用版/scripts')
from pdf_md_workflow import PDFMDWorkflow

workflow = PDFMDWorkflow(llm_client=autogen_llm_client)

def pdf_to_md(pdf_path: str) -> dict:
    return workflow.process(pdf_path)
```

### LangChain

```python
from langchain.agents import Tool
import sys
sys.path.insert(0, '/path/to/pdf-to-md通用版/scripts')
from pdf_md_workflow import PDFMDWorkflow

workflow = PDFMDWorkflow(llm_client=langchain_llm)
tools = [Tool(name="PDF转MD", func=lambda x: workflow.process(x))]
```

## 环境变量说明

| 变量名 | 必填 | 说明 |
|--------|------|------|
| MINERU_API_KEY | 是 | PDF 提取服务 API Key |
| MINIMAX_API_KEY | 是* | MiniMax M2.7 API Key（*使用 MiniMax 时必填） |
| ANTHROPIC_API_KEY | 是* | Claude API Key（*使用 Claude 时必填） |
| OPENAI_API_KEY | 是* | GPT API Key（*使用 GPT 时必填） |
| NVIDIA_API_KEY | 是* | NVIDIA DeepSeek API Key（*使用 NVIDIA 时必填） |
| IMAGE_HOST_TOKEN | 否 | 图床上传 Token |

## API 地址

- **Mineru**: `https://mineru.net`
- **MiniMax**: `https://api.minimaxi.com/v1`
- **Anthropic**: `https://api.anthropic.com`
- **OpenAI**: `https://api.openai.com/v1`
- **NVIDIA**: `https://integrate.api.nvidia.com/v1`

## 优化规则

LLM 优化包含以下通用规则：

1. **标题层级修复** - 正确识别章节层级，去除页码后缀
2. **公式处理** - 公式块原样保留，行内公式智能处理
3. **表格转换** - 简单表格转 Markdown，复杂表格保留 HTML
4. **代码块处理** - 保留原始代码和语言标识
5. **格式清理** - 去除多余空行，修复乱码，统一空格

## 返回值

```python
result = workflow.process("document.pdf")
# {
#     "md_path": "output/document.md",        # 原始提取的 MD
#     "optimized_path": "output/document_优化.md",  # 优化后的 MD
#     "extracted": True,                      # 是否进行了提取
#     "images_count": 5,                      # 图片数量
#     "uploaded": True                        # 是否上传了图床
# }
```

## License

MIT
