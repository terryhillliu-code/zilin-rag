#!/usr/bin/env python3
"""将 NotebookLM 报告保存到 Obsidian Vault

使用：
    python scripts/save_to_obsidian.py --file ~/Desktop/report.md --topic "MoE架构"
    python scripts/save_to_obsidian.py --file ~/Desktop/report.md --topic "MoE架构" --category "AI系统"
"""
import argparse
import shutil
from pathlib import Path
from datetime import datetime


def save_to_obsidian(source_file: str, topic: str, category: str = "研究报告") -> Path:
    """将文件保存到 Obsidian Vault"""
    source = Path(source_file).expanduser()

    if not source.exists():
        print(f"❌ 源文件不存在: {source}")
        return None

    # 目标目录
    vault = Path("~/Documents/ZhiweiVault").expanduser()
    target_dir = vault / category

    # 创建目录（如果不存在）
    target_dir.mkdir(parents=True, exist_ok=True)

    # 目标文件名
    date_str = datetime.now().strftime("%Y-%m-%d")
    target_name = f"PAPER_{date_str}_{topic}.md"
    target_file = target_dir / target_name

    # 复制文件
    shutil.copy2(source, target_file)

    print(f"✅ 已保存到 Obsidian: {target_file}")
    return target_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="保存报告到 Obsidian")
    parser.add_argument("--file", type=str, required=True, help="源文件路径")
    parser.add_argument("--topic", type=str, required=True, help="研究主题")
    parser.add_argument("--category", type=str, default="研究报告", help="目标目录")
    args = parser.parse_args()

    result = save_to_obsidian(args.file, args.topic, args.category)
    if result:
        print(f"\n🎉 完成! 文件路径: {result}")