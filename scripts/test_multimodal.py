#!/usr/bin/env python3
"""
多模态能力测试脚本

验证 PPT 解析和图片处理功能
"""
import os
import sys
import argparse
import json
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """检查依赖是否安装"""
    deps = {
        "python-pptx": "pptx",
        "Pillow": "PIL",
        "paddleocr": "paddleocr",
        "paddlepaddle": "paddle"
    }

    results = {}
    for name, module in deps.items():
        try:
            __import__(module)
            results[name] = {"installed": True, "version": None}
            # 尝试获取版本
            try:
                mod = __import__(module)
                if hasattr(mod, "__version__"):
                    results[name]["version"] = mod.__version__
            except:
                pass
        except ImportError:
            results[name] = {"installed": False}

    return results


def check_ocr():
    """检查 OCR 可用性"""
    try:
        from multimodal.ocr_engine import check_ocr_availability
        return check_ocr_availability()
    except ImportError as e:
        return {"available": False, "error": str(e)}


def check_vlm():
    """检查 VLM 可用性"""
    try:
        from multimodal.vlm_engine import check_vlm_availability
        return check_vlm_availability()
    except ImportError as e:
        return {"available": False, "error": str(e)}


def test_ppt_parser(file_path: str):
    """测试 PPT 解析"""
    from ingest.ppt_parser import PPTParser

    parser = PPTParser(extract_images=True)
    doc = parser.parse(file_path)

    print(f"\n{'='*60}")
    print(f"PPT 解析结果: {doc.filename}")
    print(f"{'='*60}")
    print(f"幻灯片数: {doc.total_slides}")
    print(f"图片数: {doc.total_images}")

    if doc.metadata:
        print(f"\n元数据:")
        for k, v in doc.metadata.items():
            if v:
                print(f"  {k}: {v}")

    print(f"\n幻灯片预览:")
    for slide in doc.slides[:3]:
        print(f"\n  [{slide.slide_num}] {slide.title or '(无标题)'}")
        if slide.content:
            content_preview = slide.content[0][:100] if slide.content else ""
            print(f"      内容: {content_preview}...")
        if slide.images:
            print(f"      图片: {len(slide.images)} 张")

    return doc


def test_image_processor(image_path: str):
    """测试图片处理"""
    from ingest.image_processor import ImageProcessor

    processor = ImageProcessor(prefer_local=True)
    result = processor.process_image(image_path)

    print(f"\n{'='*60}")
    print(f"图片处理结果: {Path(image_path).name}")
    print(f"{'='*60}")
    print(f"图片类型: {result.image_type}")
    print(f"处理时间: {result.processing_time:.2f}s")
    print(f"使用模型: {result.models_used}")

    if result.ocr_text:
        print(f"\nOCR 文字 (前 200 字符):")
        print(f"  {result.ocr_text[:200]}...")

    if result.vlm_description:
        print(f"\nVLM 描述 (前 200 字符):")
        print(f"  {result.vlm_description[:200]}...")

    return result


def test_image_extractor(file_path: str):
    """测试图片提取"""
    from multimodal.image_extractor import ImageExtractor

    extractor = ImageExtractor()
    images = extractor.extract(file_path)

    print(f"\n{'='*60}")
    print(f"图片提取结果: {Path(file_path).name}")
    print(f"{'='*60}")
    print(f"提取图片数: {len(images)}")

    for img in images[:5]:
        print(f"  - {Path(img.path).name} ({img.width}x{img.height})")

    return images


def main():
    parser = argparse.ArgumentParser(description="多模态能力测试")
    parser.add_argument("--check", action="store_true", help="检查依赖和可用性")
    parser.add_argument("--ppt", help="测试 PPT 解析")
    parser.add_argument("--image", help="测试图片处理")
    parser.add_argument("--extract", help="从 PDF/PPT 提取图片")
    parser.add_argument("--status", action="store_true", help="查看处理器状态")
    args = parser.parse_args()

    # 默认执行检查
    if not any([args.check, args.ppt, args.image, args.extract, args.status]):
        args.check = True

    if args.check:
        print("\n" + "="*60)
        print("依赖检查")
        print("="*60)

        deps = check_dependencies()
        for name, info in deps.items():
            status = "✅" if info["installed"] else "❌"
            version = f" (v{info['version']})" if info.get("version") else ""
            print(f"  {status} {name}{version}")

        print("\n" + "="*60)
        print("OCR 可用性")
        print("="*60)
        ocr_status = check_ocr()
        print(f"  可用: {'✅' if ocr_status.get('available') else '❌'}")
        print(f"  引擎: {ocr_status.get('engine', 'N/A')}")
        print(f"  设备: {ocr_status.get('device', 'N/A')}")
        if ocr_status.get("error"):
            print(f"  错误: {ocr_status['error']}")

        print("\n" + "="*60)
        print("VLM 可用性")
        print("="*60)
        vlm_status = check_vlm()
        print(f"  本地模型: {'✅' if vlm_status.get('local_available') else '❌'}")
        print(f"  云端 API: {'✅' if vlm_status.get('cloud_available') else '❌'}")
        print(f"  设备: {vlm_status.get('device', 'N/A')}")

    if args.ppt:
        test_ppt_parser(args.ppt)

    if args.image:
        test_image_processor(args.image)

    if args.extract:
        test_image_extractor(args.extract)

    if args.status:
        from ingest.image_processor import ImageProcessor
        processor = ImageProcessor()
        status = processor.get_status()
        print("\n" + "="*60)
        print("处理器状态")
        print("="*60)
        print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()