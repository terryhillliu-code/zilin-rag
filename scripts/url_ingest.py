"""
网页情报收录器 (v5.3)
- 目的：将网页内容转化为知识库中的“情报”
- 流程：Fetch (WebReader) -> Save (Obsidian Markdown) -> Index (ChromaDB)
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.web_reader import get_web_markdown
from ingest.distill_template import generate_distill_prompt
import requests
import json

# 加载系统密钥
sys.path.insert(0, str(Path.home() / "scripts"))
try:
    from load_secrets import load_secrets
    load_secrets(silent=True)
except ImportError:
    pass

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DASHSCOPE_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 配置
VAULT_INTEL_ROOT = Path("~/Documents/ZhiweiVault/Inbox/Intelligence").expanduser()

def url_ingest(url: str, title: Optional[str] = None, distill: bool = False):
    """收录一个 URL 并存入 Obsidian"""
    if not VAULT_INTEL_ROOT.exists():
        VAULT_INTEL_ROOT.mkdir(parents=True, exist_ok=True)
        
    # 1. 抓取内容
    content = get_web_markdown(url)
    if not content:
        print(f"❌ 无法从 {url} 获取内容")
        return False
        
    # 2. 准备文件名与元数据
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # 如果没有标题，从内容中提取第一行或使用 URL hash
    if not title:
        first_line = content.split('\n')[0].strip('# ')
        title = first_line[:50] if first_line else f"Intel_{timestamp}"
    
    # 清理标题非法字符
    safe_title = "".join([c for c in title if c.isalnum() or c in (" ", "-", "_")]).strip()
    filename = f"INTEL_{timestamp}_{safe_title}.md"
    file_path = VAULT_INTEL_ROOT / filename
    
    if distill and DASHSCOPE_API_KEY:
        print(f"🧠 正在执行[博主式提炼] (直连 DashScope)...", file=sys.stderr)
        prompt = generate_distill_prompt({"title": title, "source_url": url}, content)
        try:
            resp = requests.post(
                DASHSCOPE_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen-plus",
                    "messages": [
                        {"role": "system", "content": "你是一位顶级的深度技术博主，擅长将复杂技术转化为洞察力。"},
                        {"role": "user", "content": prompt + "\n\n内容如下：\n" + content[:10000]}
                    ],
                    "temperature": 0.3
                },
                timeout=60
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # 标记标签
                tags = ["#intel_hub", "#distilled", "#blogger_style"]
            else:
                print(f"⚠️ 提炼失败 (HTTP {resp.status_code})，回退到原始抓取")
                print(f"响应: {resp.text[:200]}", file=sys.stderr)
                tags = ["#intel_hub", "#web_clipper"]
        except Exception as e:
            print(f"⚠️ 提炼出错: {e}")
            tags = ["#intel_hub", "#web_clipper"]
    elif distill:
        print(f"⚠️ 缺少 DASHSCOPE_API_KEY，跳过提炼")
        tags = ["#intel_hub", "#web_clipper"]
    else:
        tags = ["#intel_hub", "#web_clipper"]

    # 4. 注入元数据 Frontmatter
    frontmatter = f"""---
title: "{title}"
source_url: "{url}"
captured_at: "{datetime.now().isoformat()}"
tags: {json.dumps(tags)}
status: "processed"
---

"""
    full_content = frontmatter + content
    
    # 4. 写入文件
    try:
        file_path.write_text(full_content, encoding='utf-8')
        print(f"✅ 情报已存入 Obsidian: {file_path.name}")
        
        # 5. 提示用户（或者自动触发同步）
        print(f"💡 稍后运行 reconcile_obsidian.py 即可使其进入 ChromaDB 检索。")
        return True
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zhiwei URL Ingestor")
    parser.add_argument("url", help="Target URL to ingest")
    parser.add_argument("--title", help="Optional title for the note")
    
    parser.add_argument("--distill", action="store_true", help="Enable intelligent blogger-style distillation")
    
    args = parser.parse_args()
    url_ingest(args.url, args.title, args.distill)
