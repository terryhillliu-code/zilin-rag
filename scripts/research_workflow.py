#!/usr/bin/env python3
"""自动化研究工作流

一键完成：检索 → 合并 → 上传 → 生成 → 下载

使用：
    python scripts/research_workflow.py --topic "MoE架构"
    python scripts/research_workflow.py --topic "Transformer" --dry-run
"""
import argparse
import os
import subprocess
import sys
import tempfile
import shlex
from pathlib import Path

# 离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))
from api import retrieve


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """执行命令并返回结果"""
    print(f"  执行: {' '.join(cmd[:3])}...")
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=check)
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 命令失败: {e.stderr[:200] if e.stderr else str(e)}")
        raise


def research_workflow(topic: str, top_k: int = 50, dry_run: bool = False) -> Path:
    """执行完整研究工作流"""
    print(f"\n🚀 研究工作流: {topic}")
    print(f"   top_k={top_k}, dry_run={dry_run}\n")

    # ===== Phase 1: 检索相关文档 =====
    print("[Phase 1] 检索相关文档...")
    results = retrieve(topic, top_k=top_k)

    if not results:
        print("❌ 未找到相关文档")
        sys.exit(1)

    sources = [getattr(r, 'source', '') for r in results]
    print(f"✅ 检索到 {len(results)} 条结果，来自 {len(set(sources))} 个文件")

    if dry_run:
        print("\n[DRY-RUN] 预览相关源文件:")
        for src in sorted(set(sources))[:10]:
            print(f"  - {Path(src).name}")
        print("\n✅ DRY-RUN 完成，未执行后续步骤")
        return Path("/tmp/dry_run.md")

    # ===== Phase 2: 合并笔记 =====
    print("\n[Phase 2] 合并笔记...")

    # 直接使用检索结果生成小文件（NotebookLM 大文件上传有限制）
    combined_file = tempfile.mktemp(suffix=".md", prefix="research_")

    print("  使用检索结果生成源文件...")
    with open(combined_file, 'w') as f:
        f.write(f"# {topic}\n\n")
        f.write("## 相关文档片段\n\n")
        for i, r in enumerate(results[:30]):  # 限制 30 条避免过大
            text = getattr(r, 'text', '')[:800]
            src = getattr(r, 'source', '')
            f.write(f"### [{i+1}] {Path(src).name}\n\n{text}\n\n")

    size = Path(combined_file).stat().st_size / 1024
    print(f"✅ 源文件生成完成: {combined_file} ({size:.1f} KB)")

    # ===== Phase 3: NotebookLM 处理 =====
    print("\n[Phase 3] NotebookLM 处理...")

    # 创建 Notebook
    create_cmd = [
        "bash", "-c",
        f"source ~/zhiwei-shared-venv/bin/activate && notebooklm create '{topic}'"
    ]
    result = run_cmd(create_cmd)

    # 提取 notebook ID
    output = result.stdout
    if "Created notebook:" in output:
        notebook_id = output.split("Created notebook:")[1].split("-")[0].strip()
    else:
        # 从 list 获取最新 notebook
        print("  从列表获取 notebook ID...")
        list_result = run_cmd([
            "bash", "-c",
            "source ~/zhiwei-shared-venv/bin/activate && notebooklm list"
        ])
        # 解析表格输出，提取第一个 ID
        lines = list_result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith('│') and '│' in line[10:]:
                parts = [p.strip() for p in line.split('│')]
                if parts and len(parts) > 1 and parts[1] and not parts[1].startswith('ID'):
                    notebook_id = parts[1]
                    if len(notebook_id) > 8:
                        notebook_id = notebook_id[:8]  # 取前8位
                    break

    if not notebook_id:
        print("❌ Notebook 创建失败，无法获取 notebook ID")
        Path(combined_file).unlink(missing_ok=True)
        sys.exit(1)

    print(f"✅ Notebook 创建成功: {notebook_id}")

    try:
        # 上传源文件
        print("  上传源文件...")
        safe_topic = shlex.quote(topic)
        run_cmd([
            "bash", "-c",
            f"source ~/zhiwei-shared-venv/bin/activate && notebooklm use {notebook_id}"
        ])
        run_cmd([
            "bash", "-c",
            f"source ~/zhiwei-shared-venv/bin/activate && notebooklm source add {combined_file} --title {safe_topic}"
        ])

        # 等待处理
        print("  等待源文件处理...")
        run_cmd([
            "bash", "-c",
            "source ~/zhiwei-shared-venv/bin/activate && notebooklm source wait"
        ])

        # 生成报告
        print("  生成研究报告...")
        run_cmd([
            "bash", "-c",
            f"source ~/zhiwei-shared-venv/bin/activate && notebooklm generate report --format study-guide --language zh_Hans --wait"
        ])

        # 下载报告
        print("\n[Phase 4] 下载报告...")
        output_dir = Path("~/Documents/ZhiweiVault/研究报告").expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{topic}.md"

        safe_output = shlex.quote(str(output_file))
        run_cmd([
            "bash", "-c",
            f"source ~/zhiwei-shared-venv/bin/activate && notebooklm download report {safe_output}"
        ])

        print(f"✅ 报告已保存: {output_file}")

        # 清理临时文件
        Path(combined_file).unlink(missing_ok=True)

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"❌ NotebookLM 处理失败: {e}")
        Path(combined_file).unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化研究工作流")
    parser.add_argument("--topic", type=str, required=True, help="研究主题")
    parser.add_argument("--top-k", type=int, default=50, help="检索数量")
    parser.add_argument("--dry-run", action="store_true", help="仅预览不执行")
    args = parser.parse_args()

    result = research_workflow(args.topic, args.top_k, args.dry_run)
    print(f"\n🎉 工作流完成! 输出: {result}")