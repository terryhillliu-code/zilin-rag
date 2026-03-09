#!/usr/bin/env python3
"""
模型评估测试脚本
测试 8 个模型在 Claude Code CLI 场景下的表现
"""
import subprocess
import time
import json
import os
from pathlib import Path
from datetime import datetime

# 测试模型列表
MODELS = [
    "qwen3.5-plus",
    "qwen3-max-2026-01-23",
    "qwen3-coder-next",
    "qwen3-coder-plus",  # 基准
    "glm-5",
    "glm-4.7",
    "kimi-k2.5",
    "MiniMax-M2.5"
]

# 测试输出目录
OUTPUT_DIR = Path.home() / "zhiwei-rag" / "scripts"
REPORT_PATH = Path.home() / "zhiwei-docs" / "analysis" / "model_eval_20260309.md"

# 测试结果记录
results = []

def run_test(model: str, task_id: int, prompt: str) -> dict:
    """执行单个测试任务"""
    print(f"\n{'='*60}")
    print(f"测试：模型={model}, 任务={task_id}")
    print(f"Prompt: {prompt[:80]}...")
    
    start_time = time.time()
    
    cmd = ["claude", "-p", prompt, "--model", model]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 分钟超时
        )
        elapsed = time.time() - start_time
        
        return {
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "elapsed": 300,
            "stdout": "",
            "stderr": "Timeout expired",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "elapsed": time.time() - start_time,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }

def verify_file_exists(filepath: str) -> bool:
    """验证文件是否存在"""
    return os.path.exists(filepath)

def verify_python_syntax(filepath: str) -> bool:
    """验证 Python 语法"""
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", filepath],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except:
        return False

def main():
    print(f"开始模型评估测试")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"报告路径：{REPORT_PATH}")
    print(f"测试模型数：{len(MODELS)}")
    
    # 任务定义
    tasks = [
        {
            "id": 1,
            "prompt_template": "创建测试文件 {output_path}，包含一个函数 get_model_name() 返回模型名称 '{model}'，使用中文注释，遵循 PEP8 规范",
            "verify_file": lambda m: OUTPUT_DIR / f"model_test_{m}.py"
        },
        {
            "id": 2,
            "prompt_template": "创建工具函数文件 {output_path}，包含 get_model_info() 和 verify_model() 两个函数，使用中文注释",
            "verify_file": lambda m: OUTPUT_DIR / f"model_util_{m}.py"
        },
        {
            "id": 3,
            "prompt_template": "分析 ~/zhiwei-bot/command_handler.py 中是否有未处理的异常，输出分析报告到 {output_path}，只读分析不修改源码",
            "verify_file": lambda m: OUTPUT_DIR / f"model_analysis_{m}.md"
        }
    ]
    
    all_results = []
    
    for model in MODELS:
        model_result = {
            "model": model,
            "tasks": []
        }
        
        for task in tasks:
            output_path = str(task["verify_file"](model))
            prompt = task["prompt_template"].format(output_path=output_path, model=model)
            
            # 执行测试
            test_result = run_test(model, task["id"], prompt)
            
            # 自动验证
            file_exists = verify_file_exists(output_path)
            syntax_ok = False
            if task["id"] in [1, 2] and file_exists:
                syntax_ok = verify_python_syntax(output_path)
            
            task_result = {
                "task_id": task["id"],
                "success": test_result["success"],
                "elapsed": round(test_result["elapsed"], 2),
                "file_exists": file_exists,
                "syntax_ok": syntax_ok,
                "error": test_result["stderr"][:200] if test_result["stderr"] else ""
            }
            
            model_result["tasks"].append(task_result)
            
            print(f"  任务{task['id']}: {'✅' if test_result['success'] else '❌'}, "
                  f"耗时{task_result['elapsed']}s, "
                  f"文件存在={'✅' if file_exists else '❌'}, "
                  f"语法正确={'✅' if syntax_ok else '❌'}")
        
        all_results.append(model_result)
    
    # 生成报告
    generate_report(all_results)
    
    print(f"\n{'='*60}")
    print(f"测试完成！报告已生成：{REPORT_PATH}")

def generate_report(all_results):
    """生成 Markdown 测试报告"""
    report = []
    report.append("# 模型评估测试报告")
    report.append("")
    report.append(f"> 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("> 测试工具：Claude Code CLI (claude -p)")
    report.append("> 测试模型数：8")
    report.append("")
    
    # 总体排名表
    report.append("## 一、总体排名")
    report.append("")
    report.append("| 排名 | 模型 | 任务 1 | 任务 2 | 任务 3 | 总耗时 | 综合评分 |")
    report.append("|------|------|--------|--------|--------|--------|----------|")
    
    # 计算评分
    scored = []
    for r in all_results:
        task_scores = []
        total_time = 0
        for t in r["tasks"]:
            score = 0
            if t["success"]: score += 40
            if t["file_exists"]: score += 25
            if t["syntax_ok"]: score += 20
            if t["elapsed"] < 60: score += 10
            elif t["elapsed"] < 120: score += 5
            task_scores.append(score)
            total_time += t["elapsed"]
        
        avg_score = sum(task_scores) / len(task_scores)
        scored.append({
            "model": r["model"],
            "tasks": r["tasks"],
            "total_time": round(total_time, 2),
            "score": round(avg_score, 1)
        })
    
    # 按评分排序
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    for i, s in enumerate(scored):
        report.append(f"| {i+1} | {s['model']} | "
                     f"{'✅' if s['tasks'][0]['success'] else '❌'} | "
                     f"{'✅' if s['tasks'][1]['success'] else '❌'} | "
                     f"{'✅' if s['tasks'][2]['success'] else '❌'} | "
                     f"{s['total_time']}s | {s['score']} |")
    
    report.append("")
    
    # 详细结果
    report.append("## 二、详细结果")
    report.append("")
    
    for s in scored:
        report.append(f"### {s['model']} (评分：{s['score']})")
        report.append("")
        report.append("| 任务 | 状态 | 耗时 | 文件存在 | 语法正确 |")
        report.append("|------|------|------|----------|----------|")
        for i, t in enumerate(s["tasks"]):
            report.append(f"| 任务{i+1} | {'✅' if t['success'] else '❌'} | "
                         f"{t['elapsed']}s | {'✅' if t['file_exists'] else '❌'} | "
                         f"{'✅' if t['syntax_ok'] else '❌'} |")
            if t["error"]:
                report.append(f"> 错误：{t['error']}")
        report.append("")
    
    # 结论
    report.append("## 三、结论与建议")
    report.append("")
    report.append("### 推荐配置")
    report.append("")
    if scored:
        best = scored[0]
        report.append(f"**主力模型**: {best['model']} (评分：{best['score']})")
        report.append("")
        report.append("**理由**:")
        report.append(f"- 任务完成度：{sum(1 for t in best['tasks'] if t['success'])}/3")
        report.append(f"- 平均耗时：{best['total_time']/3:.1f}s")
        report.append("")
        
        if len(scored) > 1:
            report.append(f"**备选模型**: {scored[1]['model']} (评分：{scored[1]['score']})")
    
    report.append("")
    report.append("---")
    report.append("*报告由 model_eval_runner.py 自动生成*")
    
    # 写入文件
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"报告已写入：{REPORT_PATH}")

if __name__ == "__main__":
    main()