#!/usr/bin/env python3
"""将 NotebookLM 报告保存到 Obsidian Vault

使用：
    python scripts/save_to_obsidian.py --file ~/Desktop/report.md --topic "MoE架构"
    python scripts/save_to_obsidian.py --file ~/Desktop/report.md --topic "MoE架构" --category "AI系统"
"""
import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime


def save_to_obsidian(source_file: str, topic: str, category: str = "研究报告", overwrite: bool = False) -> Path | None:
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
    base_name = f"PAPER_{date_str}_{topic}"
    target_file = target_dir / f"{base_name}.md"

    # 处理文件冲突
    if target_file.exists():
        if overwrite:
            print(f"⚠️ 覆盖已存在的文件: {target_file}")
        else:
            # 添加版本号
            version = 2
            while target_file.exists():
                target_file = target_dir / f"{base_name}_v{version}.md"
                version += 1
            print(f"⚠️ 文件已存在，使用新名称: {target_file.name}")

    # 复制文件
    shutil.copy2(source, target_file)

    print(f"✅ 已保存到 Obsidian: {target_file}")
    return target_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="保存报告到 Obsidian")
    parser.add_argument("--file", type=str, required=True, help="源文件路径")
    parser.add_argument("--topic", type=str, required=True, help="研究主题")
    parser.add_argument("--category", type=str, default="研究报告", help="目标目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的文件")
    args = parser.parse_args()

    result = save_to_obsidian(args.file, args.topic, args.category, args.overwrite)
    if result:
        print(f"\n🎉 完成! 文件路径: {result}")
    else:
        sys.exit(1)