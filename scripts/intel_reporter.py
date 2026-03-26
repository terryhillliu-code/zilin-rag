"""
知微情报简报生成器 (v5.5)
- 扫描 Obsidian Intelligence 目录中的最近情报
- 利用 LLM 将散点情报串联为一份深度总结
- 输出到 Reports 目录并提示用户
"""
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# 配置
VAULT_INTEL_ROOT = Path("~/Documents/ZhiweiVault/Inbox/Intelligence").expanduser()
REPORT_OUTPUT_DIR = Path("~/Documents/ZhiweiVault/70-79_个人笔记_Personal/79_周期复盘_Reviews").expanduser()

# 加载系统密钥
sys.path.insert(0, str(Path.home() / "scripts"))
try:
    from load_secrets import load_secrets
    load_secrets(silent=True)
except ImportError:
    pass

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DASHSCOPE_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def get_recent_intel(days: int = 7):
    """获取最近几天的情报内容"""
    cutoff = datetime.now() - timedelta(days=days)
    intel_items = []
    
    if not VAULT_INTEL_ROOT.exists():
        return []

    for f in VAULT_INTEL_ROOT.glob("*.md"):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime >= cutoff:
            # 读取内容并提取核心观点 (简易版)
            content = f.read_text(encoding='utf-8')
            # 提取文件名作为标题
            title = f.stem.replace("INTEL_", "")
            intel_items.append(f"### {title}\n{content[:1000]}...") # 截断以节省 token
            
    return intel_items

def generate_report(days: int = 7):
    """生成周期性简报"""
    items = get_recent_intel(days)
    if not items:
        print("📭 最近无新情报，无需生成简报。")
        return

    print(f"🧐 正在分析本周收录的 {len(items)} 条情报...")
    
    # 构建 Prompt
    system_prompt = "你是一位顶级行业分析师，擅长从散点技术情报中发现底层趋势和关联性。"
    user_prompt = f"""
请基于以下本周采集的技术情报，生成一份《知微技术情报周报 ({datetime.now().strftime('%Y-%W')})》。

要求：
1. **趋势分析**：不要简单的罗列，要指出这些情报之间是否有共同的技术趋势或行业竞争态势。
2. **深度洞察**：如果这些情报反映了某个领域（如 RAG, VLM, Infra）的重大转向，请明确指出。
3. **风险/机遇**：对我们的系统架构改进有哪些启发。
4. **格式**：使用 Markdown 格式，结构清晰。

情报列表：
{chr(10).join(items)}
"""

    try:
        resp = requests.post(
            DASHSCOPE_CHAT_URL,
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen-max", # 使用最强的模型做分析
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.5
            },
            timeout=120
        )
        
        if resp.status_code == 200:
            report_content = resp.json()["choices"][0]["message"]["content"]
            
            # 保存报告
            if not REPORT_OUTPUT_DIR.exists():
                REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            filename = f"INTEL_REPORT_{datetime.now().strftime('%Y-%m-%d')}.md"
            file_path = REPORT_OUTPUT_DIR / filename
            file_path.write_text(report_content, encoding='utf-8')
            
            print(f"✅ 情报简报已生成并存入: {file_path}")
            return True
        else:
            print(f"❌ 报告生成失败 (HTTP {resp.status_code})")
            return False
    except Exception as e:
        print(f"❌ 报告生成异常: {e}")
        return False

if __name__ == "__main__":
    generate_report()
