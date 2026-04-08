#!/usr/bin/env python3
"""
Inbox Triage - 知识咀嚼机
自动处理 Inbox 中的 Markdown 文件，分类移动到目标文件夹

Usage:
    python inbox_triage.py --dry-run     # 预览模式，不实际移动
    python inbox_triage.py --limit 10    # 只处理10个文件
    python inbox_triage.py               # 全量处理
"""
import os
import re
import sys
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# 配置
INBOX_PATH = Path.home() / "Documents" / "ZhiweiVault" / "Inbox"
VAULT_PATH = Path.home() / "Documents" / "ZhiweiVault"

# 分类映射 (tag prefix -> folder)
TAG_TO_FOLDER = {
    # AI 系统 (10-19)
    "LLM": "10-19_AI系统_AI-Systems",
    "大语言模型": "10-19_AI系统_AI-Systems",
    "机器学习": "10-19_AI系统_AI-Systems",
    "深度学习": "10-19_AI系统_AI-Systems",
    "RAG": "10-19_AI系统_AI-Systems",
    "Agent": "10-19_AI系统_AI-Systems",
    "多模态": "10-19_AI系统_AI-Systems",
    "NLP": "10-19_AI系统_AI-Systems",
    "计算机视觉": "10-19_AI系统_AI-Systems",
    "视频": "10-19_AI系统_AI-Systems",
    "图像": "10-19_AI系统_AI-Systems",
    "语音": "10-19_AI系统_AI-Systems",
    "生成式": "10-19_AI系统_AI-Systems",
    "扩散模型": "10-19_AI系统_AI-Systems",
    "强化学习": "10-19_AI系统_AI-Systems",

    # AI 硬件 (20-29)
    "GPU": "20-29_AI硬件_AI-Hardware",
    "芯片": "20-29_AI硬件_AI-Hardware",
    "加速器": "20-29_AI硬件_AI-Hardware",
    "互连": "20-29_AI硬件_AI-Hardware",
    "存储": "20-29_AI硬件_AI-Hardware",

    # 基础设施 (30-39)
    "数据中心": "30-39_基础设施_Infra-Compute",
    "服务器": "30-39_基础设施_Infra-Compute",
    "HPC": "30-39_基础设施_Infra-Compute",
    "云计算": "30-39_基础设施_Infra-Compute",

    # 网络 (40-49)
    "网络": "40-49_网络与互联_Networking",
    "光互连": "40-49_网络与互联_Networking",

    # 行业研究 (50-59)
    "行业报告": "50-59_行业研究_Industry",
    "市场分析": "50-59_行业研究_Industry",

    # 个人笔记 (70-79)
    "笔记": "70-79_个人笔记_Personal",
    "学习": "70-79_个人笔记_Personal",

    # 归档 (90-99)
    "新闻": "90-99_系统与归档_System",
}

# 默认目标
DEFAULT_FOLDER = "90-99_系统与归档_System"


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """解析 YAML frontmatter"""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        fm = yaml.safe_load(parts[1])
        body = parts[2]
        return fm or {}, body
    except yaml.YAMLError:
        return {}, content


def extract_tags(frontmatter: dict) -> list:
    """提取标签"""
    tags = []

    # 从 tags 字段提取
    if "tags" in frontmatter:
        t = frontmatter["tags"]
        if isinstance(t, list):
            tags.extend(t)
        elif isinstance(t, str):
            tags.append(t)

    # 从 title 提取关键词
    if "title" in frontmatter:
        title = frontmatter["title"]
        # 提取括号内的关键词
        brackets = re.findall(r'[（(]([^）)]+)[）)]', title)
        tags.extend(brackets)

    return tags


def classify_file(frontmatter: dict, content: str) -> str:
    """根据标签和内容分类文件"""
    tags = extract_tags(frontmatter)

    # 按标签匹配
    for tag in tags:
        tag_lower = tag.lower()
        for keyword, folder in TAG_TO_FOLDER.items():
            if keyword.lower() in tag_lower or tag_lower in keyword.lower():
                return folder

    # 按内容关键词匹配
    content_lower = content[:2000].lower()
    for keyword, folder in TAG_TO_FOLDER.items():
        if keyword.lower() in content_lower:
            return folder

    # 按文件名前缀
    filename = frontmatter.get("title", "")
    if filename.startswith("PAPER"):
        # 论文默认归入 AI 系统
        return "10-19_AI系统_AI-Systems"
    if filename.startswith("NEWS"):
        return "90-99_系统与归档_System"

    return DEFAULT_FOLDER


def process_file(filepath: Path, dry_run: bool = False) -> dict:
    """处理单个文件"""
    result = {
        "file": filepath.name,
        "status": "skip",
        "target": None,
        "reason": ""
    }

    try:
        content = filepath.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)

        # 检查是否已处理
        rag_indexed = frontmatter.get("rag_indexed", False)
        analyzed = frontmatter.get("analyzed", False)

        if not rag_indexed and not analyzed:
            result["status"] = "need_process"
            result["reason"] = "未索引，需要处理"
            return result

        # 分类
        target_folder = classify_file(frontmatter, body)
        result["target"] = target_folder

        # 检查目标是否存在
        target_path = VAULT_PATH / target_folder
        if not target_path.exists():
            result["status"] = "error"
            result["reason"] = f"目标文件夹不存在: {target_folder}"
            return result

        # 目标文件路径
        target_file = target_path / filepath.name

        # 检查是否已存在
        if target_file.exists():
            result["status"] = "duplicate"
            result["reason"] = f"目标已存在同名文件"
            return result

        # 移动文件
        if not dry_run:
            shutil.move(str(filepath), str(target_file))

        result["status"] = "moved"
        result["reason"] = f"已移动到 {target_folder}"

    except Exception as e:
        result["status"] = "error"
        result["reason"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Inbox Triage - 知识咀嚼机")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际移动文件")
    parser.add_argument("--limit", type=int, default=0, help="限制处理文件数量 (0=全部)")
    args = parser.parse_args()

    print("=" * 60)
    print("Inbox Triage - 知识咀嚼机")
    print("=" * 60)
    print(f"Inbox 路径: {INBOX_PATH}")
    print(f"模式: {'预览' if args.dry_run else '执行'}")
    print()

    # 获取所有 md 文件
    md_files = list(INBOX_PATH.glob("*.md"))
    total = len(md_files)

    if args.limit > 0:
        md_files = md_files[:args.limit]

    print(f"待处理文件: {len(md_files)} / {total}")
    print()

    # 统计
    stats = {
        "moved": 0,
        "duplicate": 0,
        "need_process": 0,
        "error": 0,
        "skip": 0,
    }

    # 处理
    for i, filepath in enumerate(md_files, 1):
        result = process_file(filepath, args.dry_run)

        # 统计
        stats[result["status"]] = stats.get(result["status"], 0) + 1

        # 输出
        status_icon = {
            "moved": "✅",
            "duplicate": "⚠️",
            "need_process": "📝",
            "error": "❌",
            "skip": "⏭️",
        }.get(result["status"], "❓")

        print(f"[{i}/{len(md_files)}] {status_icon} {result['file'][:40]}")
        if result["target"]:
            print(f"         → {result['target']}")
        if result["reason"]:
            print(f"         {result['reason']}")

    # 汇总
    print()
    print("=" * 60)
    print("处理汇总:")
    print(f"  ✅ 已移动: {stats['moved']}")
    print(f"  ⚠️  重复: {stats['duplicate']}")
    print(f"  📝 待处理: {stats['need_process']}")
    print(f"  ❌ 错误: {stats['error']}")

    if args.dry_run:
        print()
        print("⚠️  预览模式，未实际移动文件")
        print("   移除 --dry-run 参数执行实际移动")


if __name__ == "__main__":
    main()