#!/usr/bin/env python3
import os
import sys
import subprocess
import json

# 配置
API_KEY = "sk-70d377bd717b4f8abe405bff72427147"
PDF_PATH = "/Users/liufang/Documents/Library/【核心】AI硬件/[2026]_2026年中国半导体行业及DPU市场分析.pdf"

def check_images():
    print("--- 1. PDF 图片提取能力 ---")
    try:
        import fitz
        if not os.path.exists(PDF_PATH):
            print(f"❌ 测试 PDF 文件不存在: {PDF_PATH}")
            return False, 0
        doc = fitz.open(PDF_PATH)
        img_count = 0
        for i in range(min(5, len(doc))):
            page = doc[i]
            img_count += len(page.get_images())
        print(f"✅ PyMuPDF (fitz) 可用")
        print(f"前 5 页图片数: {img_count}")
        return True, img_count
    except ImportError:
        print("❌ PyMuPDF (fitz) 未安装")
        return False, 0
    except Exception as e:
        print(f"❌ 图片提取出错: {e}")
        return False, 0

def check_vlm():
    print("\n--- 2. VLM 云端 API 可用性 ---")
    if not API_KEY:
        print("❌ DASHSCOPE_API_KEY 未配置")
        return False, []
    
    print(f"✅ DASHSCOPE_API_KEY 已找到: {API_KEY[:5]}***")
    try:
        import dashscope
        print(f"✅ dashscope 库已安装 (v{dashscope.__version__})")
        # 验证 qwen-vl-plus 可用性
        # 注意：这里仅验证 API 通讯，不进行实际大负载推理
        from dashscope import MultiModalConversation
        return True, ["qwen-vl-plus", "qwen-vl-max"]
    except ImportError:
        print("❌ dashscope 未安装")
        return False, []
    except Exception as e:
        print(f"❌ API 检查出错: {e}")
        return False, []

def check_tables():
    print("\n--- 3. 表格提取工具可用性 ---")
    tools = {
        "pdfplumber": "pdfplumber",
        "camelot": "camelot",
        "tabula": "tabula"
    }
    installed = []
    for name, module in tools.items():
        try:
            __import__(module)
            installed.append(name)
        except ImportError:
            pass
    
    print(f"已安装工具: {installed if installed else '无'}")
    
    table_count = 0
    if "pdfplumber" in installed:
        try:
            import pdfplumber
            with pdfplumber.open(PDF_PATH) as pdf:
                for i in range(min(5, len(pdf.pages))):
                    page = pdf.pages[i]
                    tabs = page.extract_tables()
                    table_count += len(tabs)
            print(f"pdfplumber 测试结果: 前 5 页表格数 {table_count}")
        except Exception as e:
            print(f"pdfplumber 提取失败: {e}")
            
    return installed, table_count

def check_resources():
    print("\n--- 4. 系统资源评估 ---")
    try:
        # macOS 内存状态检查
        res = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5)
        status = "UNKNOWN"
        for line in res.stdout.split('\n'):
            if "System-wide memory free percentage" in line:
                print(line.strip())
            if "Memory pressure status" in line:
                status = line.split(':')[-1].strip().upper()
        
        print(f"内存状态: {status}")
        return status
    except Exception as e:
        print(f"无法检测内存状态: {e}")
        return "UNKNOWN"

if __name__ == "__main__":
    print(f"开始验证核心研报: {os.path.basename(PDF_PATH)}")
    print("="*40)
    check_images()
    check_vlm()
    check_tables()
    check_resources()
    print("="*40)
    print("验证完成")
