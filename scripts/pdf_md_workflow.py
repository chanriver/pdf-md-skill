"""
PDF to MD 通用工作流主脚本
支持多 Provider LLM 优化，用于 Claude Code 及各类 AI Agent 平台

用法:
    python pdf_md_workflow.py --pdf document.pdf          # 完整流程
    python pdf_md_workflow.py --extract document.pdf      # 仅提取
    python pdf_md_workflow.py --optimize document.md      # 仅优化
    python pdf_md_workflow.py --batch "*.pdf"             # 批量处理
"""

import os
import sys
import json
import time
import glob
import shutil
import argparse
import yaml
from pathlib import Path
from typing import Literal, Optional

# ============== 配置区 ==============

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ============== LLM Provider ==============

class ExternalLLMProvider:
    """外部传入的 LLM Client 包装器

    用于复用已有的 LLM Client（如 CrewAI、AutoGen、LangChain 等平台已配置的大模型）
    """

    def __init__(self, client, model: str = "MiniMax-M2.7", provider_type: str = "minimax"):
        """
        Args:
            client: 已初始化的 LLM Client（支持 anthropic.Anthropic 或 openai.OpenAI 兼容对象）
            model: 模型名称（默认 MiniMax-M2.7）
            provider_type: Client 类型 ("anthropic" | "openai" | "minimax")
        """
        self.client = client
        self.model = model
        self.provider_type = provider_type

    def optimize(self, md_content: str, custom_rules: Optional[str] = None) -> str:
        """使用外部 Client 优化 MD 内容"""
        system_prompt = self._get_system_prompt(custom_rules)

        user_prompt = f"""请优化以下 MD 文档：

---开始---
{md_content}
---结束---

请直接返回优化后的 MD 内容，不要有任何解释。"""

        if self.provider_type == "anthropic":
            return self._optimize_anthropic(system_prompt, user_prompt)
        elif self.provider_type == "minimax":
            return self._optimize_openai(system_prompt, user_prompt)  # MiniMax 也是 OpenAI 兼容接口
        else:
            return self._optimize_openai(system_prompt, user_prompt)

    def _optimize_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """使用 Anthropic Client"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return message.content[0].text

    def _optimize_openai(self, system_prompt: str, user_prompt: str) -> str:
        """使用 OpenAI 兼容 Client（MiniMax、OpenAI 等）"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=16000
        )
        return response.choices[0].message.content

    def _get_system_prompt(self, custom_rules: Optional[str] = None) -> str:
        """获取系统提示词"""
        default_rules = """对提供的 MD 文档进行以下优化：

1. **标题层级修复** - 正确识别章节层级，去除页码后缀
2. **公式处理** - 公式块原样保留，行内公式智能处理
3. **表格转换** - 简单表格转 Markdown，复杂表格保留 HTML
4. **代码块处理** - 保留原始代码和语言标识
5. **格式清理** - 去除多余空行，修复乱码，统一空格

输出要求：只返回优化后的 MD 内容，不要任何解释。"""

        if custom_rules:
            return f"""你是一个专业的文档排版专家，负责优化 Markdown 文档。

{custom_rules}

请直接返回优化后的 MD 内容，不要有任何解释。"""
        return f"""你是一个专业的文档排版专家，负责优化 Markdown 文档。

{default_rules}

请直接返回优化后的 MD 内容，不要有任何解释。"""


class LLMProviderBase:
    """LLM Provider 基类"""

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model

    def optimize(self, md_content: str, custom_rules: Optional[str] = None) -> str:
        """优化 MD 内容"""
        raise NotImplementedError

    def get_system_prompt(self, custom_rules: Optional[str] = None) -> str:
        """获取系统提示词"""
        default_rules = """对提供的 MD 文档进行以下优化：

1. **标题层级修复** - 正确识别章节层级，去除页码后缀
2. **公式处理** - 公式块原样保留，行内公式智能处理
3. **表格转换** - 简单表格转 Markdown，复杂表格保留 HTML
4. **代码块处理** - 保留原始代码和语言标识
5. **格式清理** - 去除多余空行，修复乱码，统一空格

输出要求：只返回优化后的 MD 内容，不要任何解释。"""

        if custom_rules:
            return f"""你是一个专业的文档排版专家，负责优化 Markdown 文档。

{custom_rules}

请直接返回优化后的 MD 内容，不要有任何解释。"""
        return f"""你是一个专业的文档排版专家，负责优化 Markdown 文档。

{default_rules}

请直接返回优化后的 MD 内容，不要有任何解释。"""


class AnthropicProvider(LLMProviderBase):
    """Anthropic Claude Provider"""

    def __init__(self, api_key: str, model: str = "claude-opus-4-6", **kwargs):
        super().__init__(api_key, model)
        self.client = None

    def _get_client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client

    def optimize(self, md_content: str, custom_rules: Optional[str] = None) -> str:
        client = self._get_client()
        system_prompt = self.get_system_prompt(custom_rules)

        user_prompt = f"""请优化以下 MD 文档：

---开始---
{md_content}
---结束---

请直接返回优化后的 MD 内容，不要有任何解释。"""

        message = client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return message.content[0].text


class OpenAIProvider(LLMProviderBase):
    """OpenAI GPT Provider"""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, model)
        self.base_url = base_url or "https://api.openai.com/v1"
        self.client = None

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def optimize(self, md_content: str, custom_rules: Optional[str] = None) -> str:
        client = self._get_client()
        system_prompt = self.get_system_prompt(custom_rules)

        user_prompt = f"""请优化以下 MD 文档：

---开始---
{md_content}
---结束---

请直接返回优化后的 MD 内容，不要有任何解释。"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=16000
        )

        return response.choices[0].message.content


class MiniMaxProvider(LLMProviderBase):
    """MiniMax Provider"""

    def __init__(self, api_key: str, model: str = "MiniMax-M2.7",
                 base_url: str = "https://api.minimaxi.com/v1", **kwargs):
        super().__init__(api_key, model)
        self.base_url = base_url
        self.client = None

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def optimize(self, md_content: str, custom_rules: Optional[str] = None) -> str:
        client = self._get_client()
        system_prompt = self.get_system_prompt(custom_rules)

        user_prompt = f"""请优化以下 MD 文档：

---开始---
{md_content}
---结束---

请直接返回优化后的 MD 内容，不要有任何解释。"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=16000
        )

        return response.choices[0].message.content


class NVIDIAProvider(LLMProviderBase):
    """NVIDIA DeepSeek Provider (备用)"""

    def __init__(self, api_key: str, model: str = "deepseek-ai/deepseek-v3.2",
                 base_url: str = "https://integrate.api.nvidia.com/v1", **kwargs):
        super().__init__(api_key, model)
        self.base_url = base_url
        self.client = None

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def optimize(self, md_content: str, custom_rules: Optional[str] = None) -> str:
        client = self._get_client()
        system_prompt = self.get_system_prompt(custom_rules)

        user_prompt = f"""请优化以下 MD 文档：

---开始---
{md_content}
---结束---

请直接返回优化后的 MD 内容，不要有任何解释。"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=16000
        )

        return response.choices[0].message.content


# ============== Provider 工厂 ==============

def get_provider(provider_name: str, config: dict) -> LLMProviderBase:
    """获取 LLM Provider"""

    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "minimax": MiniMaxProvider,
        "nvidia": NVIDIAProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"不支持的 provider: {provider_name}")

    provider_config = config.get(provider_name, {})
    api_key_env = provider_config.get("api_key", "")

    # 从环境变量获取
    if api_key_env.startswith("${") and api_key_env.endswith("}"):
        env_var = api_key_env[2:-1]
        api_key = os.environ.get(env_var, "")
    else:
        api_key = api_key_env

    if not api_key:
        raise ValueError(f"未设置 {provider_name} 的 API key")

    return providers[provider_name](
        api_key=api_key,
        model=provider_config.get("model", ""),
        base_url=provider_config.get("base_url", None),
    )


# ============== MinerU API ==============

class MineruExtractor:
    """Mineru PDF 提取"""

    def __init__(self, api_key: str, base_url: str = "https://mineru.net"):
        self.api_key = api_key
        self.base_url = base_url

    def extract(self, pdf_path: str, output_dir: str, language: str = "ch") -> dict:
        """提取 PDF 返回 MD"""
        import requests

        url = f"{self.base_url}/api/v4/extract"

        with open(pdf_path, "rb") as f:
            files = {"file": (Path(pdf_path).name, f, "application/pdf")}
            data = {
                "token": self.api_key,
                "language": language,
            }

            response = requests.post(url, files=files, data=data, timeout=300)

        if response.status_code != 200:
            raise Exception(f"Mineru API 错误: {response.status_code} - {response.text}")

        result = response.json()

        if result.get("code") != 0:
            raise Exception(f"Mineru 提取失败: {result}")

        # 轮询结果
        batch_id = result.get("data", {}).get("batch_id")
        return self._poll_result(batch_id, output_dir)

    def _poll_result(self, batch_id: str, output_dir: str, max_wait: int = 600) -> dict:
        """轮询提取结果"""
        import requests
        import time

        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        start_time = time.time()

        while time.time() - start_time < max_wait:
            response = requests.get(url, timeout=30)
            result = response.json()

            if result.get("code") == 0:
                data = result.get("data", {})
                status = data.get("status")

                if status == "completed":
                    # 下载结果
                    md_url = data.get("result", {}).get("markdown_url")
                    if md_url:
                        self._download_md(md_url, output_dir)

                    # 下载图片
                    images = data.get("result", {}).get("images", [])
                    self._download_images(images, output_dir)

                    return {
                        "md_path": Path(output_dir) / "output.md",
                        "images_count": len(images),
                    }

                elif status == "failed":
                    raise Exception("Mineru 提取失败")

            time.sleep(5)

        raise Exception("Mineru 提取超时")

    def _download_md(self, url: str, output_dir: str):
        """下载 MD 文件"""
        import requests

        response = requests.get(url, timeout=60)
        output_path = Path(output_dir) / "output.md"
        output_path.write_bytes(response.content)

    def _download_images(self, images: list, output_dir: str):
        """下载图片"""
        import requests

        images_dir = Path(output_dir) / "images"
        images_dir.mkdir(exist_ok=True)

        for i, img_url in enumerate(images):
            try:
                response = requests.get(img_url, timeout=30)
                ext = Path(img_url).suffix or ".png"
                img_path = images_dir / f"image_{i}{ext}"
                img_path.write_bytes(response.content)
            except Exception:
                pass


# ============== 图床上传 ==============

class ImageHostUploader:
    """图床上传器"""

    def __init__(self, token: str, url: str = "https://imgbed.361026.xyz/upload"):
        self.token = token
        self.url = url

    def upload(self, md_path: str) -> str:
        """上传 MD 中的图片并替换链接"""
        import requests
        import re

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 查找本地图片
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(image_pattern, content)

        for alt_text, image_path in matches:
            image_path = image_path.strip()
            if image_path.startswith(("http://", "https://")):
                continue

            # 上传本地图片
            if Path(image_path).exists():
                try:
                    with open(image_path, "rb") as f:
                        files = {"file": f}
                        data = {"token": self.token}
                        response = requests.post(self.url, files=files, data=data, timeout=30)
                        result = response.json()

                        if result.get("code") == 0:
                            remote_url = result.get("data", {}).get("url", "")
                            content = content.replace(image_path, remote_url)
                except Exception:
                    pass

        # 保存修改后的 MD
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        return md_path


# ============== 主工作流 ==============

class PDFMDWorkflow:
    """PDF MD 工作流"""

    def __init__(self, config_path: Optional[str] = None,
                 llm_client=None, llm_model: Optional[str] = None,
                 llm_provider_type: str = "minimax", **kwargs):
        """初始化工作流

        Args:
            config_path: 配置文件路径
            llm_client: 外部已初始化的 LLM Client（支持 anthropic/OpenAI/MiniMax 兼容对象）
                        如果传入此参数，将优先使用，不再从配置/环境变量读取
            llm_model: 模型名称（当使用外部 client 时必填，默认 MiniMax-M2.7）
            llm_provider_type: 外部 client 类型 ("anthropic" | "openai" | "minimax")
            **kwargs: 直接传入的配置参数，会覆盖配置文件
        """
        self.config = self._load_config(config_path)
        self.config.update(kwargs)

        # 初始化 Mineru
        mineru_key = os.environ.get("MINERU_API_KEY") or self.config.get("mineru", {}).get("api_key", "")
        mineru_url = self.config.get("mineru", {}).get("base_url", "https://mineru.net")
        self.mineru = MineruExtractor(mineru_key, mineru_url) if mineru_key else None

        # 初始化图床上传
        image_host = self.config.get("image_host", {})
        image_token = image_host.get("token", "")
        image_url = image_host.get("url", "")
        self.uploader = ImageHostUploader(image_token, image_url) if image_token else None

        # 外部 LLM Client（优先使用）
        self.external_llm_client = llm_client
        self.external_llm_model = llm_model or self.config.get("default_model", "MiniMax-M2.7")
        self.external_llm_provider_type = llm_provider_type

        # 默认 Provider（当没有外部 client 时使用）
        self.default_provider = self.config.get("default_provider", "minimax")

    def _load_config(self, config_path: Optional[str]) -> dict:
        """加载配置"""
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        elif DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def process(self, pdf_path: str, output_dir: Optional[str] = None,
                provider: Optional[str] = None, upload_image: bool = True,
                custom_rules: Optional[str] = None) -> dict:
        """完整流程：PDF → 提取 → 优化 → 图床

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录
            provider: LLM Provider (anthropic | openai | minimax | nvidia)
            upload_image: 是否上传图片到图床
            custom_rules: 自定义优化规则

        Returns:
            dict: {"md_path": str, "images_count": int, ...}
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        output_dir = output_dir or str(pdf_path.parent / pdf_path.stem)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        result = {}

        # 1. 提取 PDF
        if self.mineru:
            print(f"[1/3] 提取 PDF: {pdf_path.name}")
            extract_result = self.mineru.extract(str(pdf_path), output_dir)
            md_path = extract_result.get("md_path", Path(output_dir) / "output.md")
            result["extracted"] = True
            result["images_count"] = extract_result.get("images_count", 0)
        else:
            # 查找已存在的 MD
            md_candidates = list(Path(output_dir).glob("*.md")) + list(Path(output_dir).glob("*.mdx"))
            if not md_candidates:
                raise FileNotFoundError(f"未找到 MD 文件，且未配置 Mineru API")
            md_path = md_candidates[0]
            result["extracted"] = False

        result["md_path"] = str(md_path)
        print(f"[2/3] 优化 MD: {md_path.name}")

        # 2. 优化 MD
        if self.external_llm_client:
            # 使用外部传入的 LLM Client
            llm = ExternalLLMProvider(
                client=self.external_llm_client,
                model=self.external_llm_model,
                provider_type=self.external_llm_provider_type
            )
        else:
            # 从配置/环境变量获取
            llm_provider = provider or self.default_provider
            llm = get_provider(llm_provider, self.config)

        with open(md_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        optimized_content = llm.optimize(original_content, custom_rules)

        optimized_path = md_path.parent / f"{md_path.stem}_优化.md"
        with open(optimized_path, "w", encoding="utf-8") as f:
            f.write(optimized_content)

        result["optimized_path"] = str(optimized_path)
        print(f"[3/3] 优化完成: {optimized_path.name}")

        # 3. 图床上传
        if upload_image and self.uploader:
            print("上传图片到图床...")
            self.uploader.upload(str(optimized_path))
            result["uploaded"] = True

        return result

    def extract_only(self, pdf_path: str, output_dir: Optional[str] = None) -> dict:
        """仅提取 PDF，不优化"""
        if not self.mineru:
            raise ValueError("未配置 Mineru API")

        pdf_path = Path(pdf_path)
        output_dir = output_dir or str(pdf_path.parent / pdf_path.stem)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        result = self.mineru.extract(str(pdf_path), output_dir)
        result["output_dir"] = output_dir
        return result

    def optimize_only(self, md_path: str, output_path: Optional[str] = None,
                      provider: Optional[str] = None,
                      custom_rules: Optional[str] = None) -> dict:
        """仅优化已有 MD 文件

        Args:
            md_path: MD 文件路径
            output_path: 输出路径（可选）
            provider: LLM Provider（当使用外部 client 时忽略）
            custom_rules: 自定义优化规则

        Returns:
            dict: {"optimized_path": str}
        """
        md_path = Path(md_path)
        if not md_path.exists():
            raise FileNotFoundError(f"MD 文件不存在: {md_path}")

        # 获取 LLM
        if self.external_llm_client:
            # 使用外部传入的 LLM Client
            llm = ExternalLLMProvider(
                client=self.external_llm_client,
                model=self.external_llm_model,
                provider_type=self.external_llm_provider_type
            )
        else:
            # 从配置/环境变量获取
            llm_provider = provider or self.default_provider
            llm = get_provider(llm_provider, self.config)

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        optimized_content = llm.optimize(content, custom_rules)

        output_path = output_path or str(md_path.parent / f"{md_path.stem}_优化.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(optimized_content)

        return {"optimized_path": output_path}


# ============== CLI 入口 ==============

def main():
    parser = argparse.ArgumentParser(description="PDF to MD 工作流")
    parser.add_argument("--pdf", help="PDF 文件路径")
    parser.add_argument("--extract", help="仅提取 PDF，不优化")
    parser.add_argument("--optimize", help="仅优化已有 MD 文件")
    parser.add_argument("--batch", help="批量处理，支持 glob 模式")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--provider", choices=["anthropic", "openai", "minimax", "nvidia"],
                        help="LLM Provider")
    parser.add_argument("--no-image", action="store_true", help="跳过图床上传")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--rules", help="自定义优化规则文件路径")

    args = parser.parse_args()

    # 加载配置
    config_path = args.config or str(DEFAULT_CONFIG_PATH)
    workflow = PDFMDWorkflow(config_path)

    # 加载自定义规则
    custom_rules = None
    if args.rules and Path(args.rules).exists():
        custom_rules = Path(args.rules).read_text(encoding="utf-8")

    # 确定处理模式
    if args.extract:
        # 仅提取
        result = workflow.extract_only(args.extract, args.output_dir)
        print(f"提取完成: {result}")

    elif args.optimize:
        # 仅优化
        result = workflow.optimize_only(
            args.optimize,
            output_path=args.output_dir,
            provider=args.provider,
            custom_rules=custom_rules,
        )
        print(f"优化完成: {result['optimized_path']}")

    elif args.batch:
        # 批量处理
        files = glob.glob(args.batch)
        for i, f in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] 处理: {f}")
            try:
                workflow.process(
                    f,
                    output_dir=args.output_dir,
                    provider=args.provider,
                    upload_image=not args.no_image,
                    custom_rules=custom_rules,
                )
            except Exception as e:
                print(f"错误: {e}")

    elif args.pdf:
        # 完整流程
        result = workflow.process(
            args.pdf,
            output_dir=args.output_dir,
            provider=args.provider,
            upload_image=not args.no_image,
            custom_rules=custom_rules,
        )
        print(f"\n完成!")
        print(f"  优化后 MD: {result.get('optimized_path')}")
        if result.get("images_count"):
            print(f"  图片数量: {result.get('images_count')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
