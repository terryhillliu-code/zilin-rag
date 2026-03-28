#!/usr/bin/env python3
"""
批量 PDF 入库脚本 v2.0
- 支持 OCR 自动回退（扫描件）
- 同步更新 klib.db vectorized 标志
"""
import asyncio
import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.pdf_multimodal_ingest import ingest_pdf_multimodal
from ingest.lance_store import LanceStore
from ingest.vlm_describer import VLMDescriber

# klib.db 路径
KLIB_DB = Path.home() / "Documents/Library/klib.db"


def update_klib_vectorized(file_path: str):
    """更新 klib.db 中的 vectorized 标志"""
    if not KLIB_DB.exists():
        return

    try:
        conn = sqlite3.connect(str(KLIB_DB))
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE books SET vectorized = 1, vectorized_at = unixepoch('now') WHERE file_path = ?",
            (file_path,)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"   ⚠️ klib.db 更新失败: {e}")


async def bulk_ingest(file_list_path, use_ocr=False):
    store = LanceStore()
    vlm = None  # Disable VLM for bulk speed

    with open(file_list_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"🚀 Starting Bulk Ingest of {len(files)} files...")
    print(f"   OCR 模式: {'启用' if use_ocr else '禁用（自动回退）'}")

    success_count = 0
    fail_count = 0
    ocr_count = 0

    for i, pdf_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing: {os.path.basename(pdf_path)}")
        try:
            # 第一次尝试：不启用 OCR
            result = await ingest_pdf_multimodal(
                pdf_path,
                store,
                vlm_describer=vlm,
                use_mineru=use_ocr
            )

            # 如果返回 0 chunks 且未启用 OCR，尝试启用 OCR
            if result['total'] == 0 and not use_ocr:
                print(f"   🔄 检测到 0 chunks，启用 OCR 重试...")
                result = await ingest_pdf_multimodal(
                    pdf_path,
                    store,
                    vlm_describer=vlm,
                    use_mineru=True
                )
                if result['total'] > 0:
                    ocr_count += 1

            if result['total'] > 0:
                print(f"   ✅ Done: {result['total']} chunks")
                success_count += 1
                # 更新 klib.db
                update_klib_vectorized(pdf_path)
            else:
                print(f"   ⚠️ Done: 0 chunks (可能是空 PDF)")
                success_count += 1
                update_klib_vectorized(pdf_path)

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            fail_count += 1

    print(f"\n📊 Bulk Ingest Complete!")
    print(f"   Total: {len(files)}")
    print(f"   Success: {success_count}")
    print(f"   Failed: {fail_count}")
    print(f"   OCR 处理: {ocr_count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="批量 PDF 入库 v2.0")
    parser.add_argument("file_list", help="文件列表路径")
    parser.add_argument("--ocr", action="store_true", help="强制启用 OCR 模式")
    args = parser.parse_args()

    asyncio.run(bulk_ingest(args.file_list, use_ocr=args.ocr))
