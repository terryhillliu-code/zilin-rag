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
import time
from pathlib import Path

# 离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))
from api import retrieve


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """执行命令并返回结果"""
    print(f"  执行: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


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
    combined_file = tempfile.mktemp(suffix=".md", prefix="research_")

    # 使用 obs2nlm 合整个 Vault（简化版，实际应按目录筛选）
    obs2nlm_cmd = [
        "obs2nlm",
        "--vault", "~/Documents/ZhiweiVault",
        "--source", combined_file
    ]

    # 使用 shell 展开 ~ 路径
    result = run_cmd(["bash", "-c", f"source ~/zhiwei-shared-venv/bin/activate && obs2nlm --vault ~/Documents/ZhiweiVault --source {combined_file}"])

    if Path(combined_file).exists():
        size = Path(combined_file).stat().st_size / 1024 / 1024
        print(f"✅ 合并完成: {combined_file} ({size:.1f} MB)")
    else:
        print("⚠️ 合并文件未生成，将直接使用检索文本")
        # 备用方案：直接写入检索结果
        with open(combined_file, 'w') as f:
            f.write(f"# {topic} 研究资料\n\n")
            f.write("## 相关文档片段\n\n")
            for i, r in enumerate(results[:20]):
                text = getattr(r, 'text', '')[:500]
                src = getattr(r, 'source', '')
                f.write(f"### {i+1}. {Path(src).name}\n\n{text}...\n\n")
        print(f"✅ 备用文件已生成: {combined_file}")

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
        notebook_id = None
        # 从 list 获取最新
        list_result = run_cmd(["bash", "-c", "source ~/zhiwei-shared-venv/bin/activate && notebooklm list"])
        # 解析最新 notebook

    if notebook_id:
        print(f"✅ Notebook 创建成功: {notebook_id}")

        # 上传源文件
        print("  上传源文件...")
        run_cmd(["bash", "-c", f"source ~/zhiwei-shared-venv/bin/activate && notebooklm use {notebook_id}"])
        run_cmd(["bash", "-c", f"source ~/zhiwei-shared-venv/bin/activate && notebooklm source add {combined_file} --title '{topic}研究资料'"])

        # 等待处理
        print("  等待源文件处理...")
        run_cmd(["bash", "-c", "source ~/zhiwei-shared-venv/bin/activate && notebooklm source wait"])

        # 生成报告
        print("  生成研究报告...")
        run_cmd(["bash", "-c", f"source ~/zhiwei-shared-venv/bin/activate && notebooklm generate report --format study-guide --language zh_Hans --wait"])

        # 下载报告
        print("\n[Phase 4] 下载报告...")
        output_dir = Path("~/Documents/ZhiweiVault/研究报告").expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{topic}.md"

        run_cmd(["bash", "-c", f"source ~/zhiwei-shared-venv/bin/activate && notebooklm download report {output_file}"])

        print(f"✅ 报告已保存: {output_file}")

        # 清理临时文件
        Path(combined_file).unlink(missing_ok=True)

        return output_file
    else:
        print("❌ Notebook 创建失败")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化研究工作流")
    parser.add_argument("--topic", type=str, required=True, help="研究主题")
    parser.add_argument("--top-k", type=int, default=50, help="检索数量")
    parser.add_argument("--dry-run", action="store_true", help="仅预览不执行")
    args = parser.parse_args()

    result = research_workflow(args.topic, args.top_k, args.dry_run)
    print(f"\n🎉 工作流完成! 输出: {result}")