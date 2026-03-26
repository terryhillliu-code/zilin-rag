#!/usr/bin/env python3
"""
VaultSyncMaster v3.0 ⭐ (Research V4.4)
统一的 Obsidian 笔记同步与对齐工具。
支持 LanceDB (深度研究) 与 ChromaDB (通用对等) 的全量同步与自愈。
"""
import os
import sys
import yaml
import argparse
import subprocess
import hashlib
import json
import urllib.request
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.lance_store import LanceStore

# 配置
VAULT_ROOT = Path("~/Documents/ZhiweiVault").expanduser()
EXCLUDE_DIRS = {".obsidian", "attachments", "extracted", "backup_20260315"}

# ChromaDB 配置 (用于通用搜索)
CHROMA_PATH = "/root/downloads/knowledge-library/chromadb"
CHROMA_COLLECTION = "knowledge_base"
CONTAINER_NAME = "clawdbot"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_existing_chroma_ids():
    """从 Docker 容器内的 ChromaDB 获取已存在的文档 ID"""
    try:
        # 增加超时和更详细的输出捕获
        cmd = f"""
import chromadb
import json
import sys
try:
    client = chromadb.PersistentClient(path="{CHROMA_PATH}")
    collection = client.get_or_create_collection("{CHROMA_COLLECTION}")
    results = collection.peek(limit=100000)
    ids = set(results.get("ids", []))
    for meta in results.get("metadatas", []):
        if "doc_id" in meta: ids.add(meta["doc_id"])
    print(json.dumps(list(ids)))
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "python3", "-c", cmd],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return set(json.loads(result.stdout.strip()))
        else:
            print(f"⚠️  Chroma 脚本返回错误 (Code {result.returncode}): {result.stderr.strip()}")
    except Exception as e:
        print(f"⚠️  容器访问失败 (可能正在启动中或权限不足): {e}")
    return set()

def vectorize_to_chroma(note_path: Path, doc_id: str):
    """将笔记向量化并存入 ChromaDB (通过 Docker exec)"""
    try:
        content = note_path.read_text(encoding='utf-8', errors='ignore')
        if not content.strip() or len(content.strip()) < 10:
            return False

        # 准备元数据
        metadata = {
            "doc_id": doc_id,
            "title": note_path.stem,
            "category": "obsidian_note",
            "source": str(note_path.relative_to(VAULT_ROOT)),
            "file_path": str(note_path)
        }

        # 生成向量 (使用云端 API)
        embed_data = json.dumps({
            "model": "text-embedding-v3",
            "input": content[:4096],
            "dimension": 1024
        }).encode()

        req = urllib.request.Request(
            DASHSCOPE_API_URL,
            data=embed_data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            embedding = json.loads(resp.read())["data"][0]["embedding"]

        # 写入 ChromaDB (容器内执行)
        embed_str = json.dumps(embedding)
        meta_str = json.dumps(metadata, ensure_ascii=False)
        escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', '\\n')[:4000]

        cmd = f"""
import chromadb
client = chromadb.PersistentClient(path="{CHROMA_PATH}")
collection = client.get_or_create_collection("{CHROMA_COLLECTION}")
collection.add(embeddings=[{embed_str}], metadatas=[{meta_str}], documents=["{escaped_content}"], ids=["{doc_id}"])
"""
        res = subprocess.run(["docker", "exec", CONTAINER_NAME, "python3", "-c", cmd], capture_output=True, text=True)
        return res.returncode == 0
    except Exception as e:
        print(f"  ❌ Chroma 向量化失败: {e}")
        return False

def is_arxiv_paper(filepath):
    """通过 YAML frontmatter 判断是否为 ArXiv 论文"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(1024)
            if not content.startswith('---'):
                return False
            if 'arxiv.org' in content or 'arxiv_id' in content or 'type: paper' in content:
                return True
    except:
        pass
    return False

def get_indexed_sources(store):
    """获取 LanceDB 中已存在的 source (绝对路径)"""
    try:
        if store.table is None: 
            print("⚠️  LanceDB 表不存在，返回空索引")
            return set()
        
        # 使用 to_arrow() 获取列，避免 pandas 转换和 select 接口版本差异
        data = store.table.to_arrow()
        if "source" in data.column_names:
            return set(data.column("source").to_pylist())
        return set()
    except Exception as e:
        print(f"⚠️  LanceDB 读取失败: {e}")
        return set()

def reconcile(dry_run=False, limit=0, skip_chroma=False):
    print(f"🚀 开始全量同步 (VaultSyncMaster v3.0): {VAULT_ROOT}")
    
    # 1. 初始化 LanceDB
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    store = LanceStore(db_path=config['paths']['lance_db'])
    
    # 2. 获取双库索引列表
    lance_sources = get_indexed_sources(store)
    chroma_ids = set()
    if not skip_chroma:
        chroma_ids = get_existing_chroma_ids()
    else:
        print("⏩ 跳过 ChromaDB 索引检查")
    
    print(f"📊 索引状态: LanceDB({len(lance_sources)} 来源), ChromaDB({len(chroma_ids)} 文档)")
    
    # 3. 扫描全库文件
    all_papers = []  # (path, chroma_id)
    all_notes = []   # (path, chroma_id)
    
    for md_file in VAULT_ROOT.rglob("*.md"):
        # 跳过排除项
        if any(part in EXCLUDE_DIRS for part in md_file.parts):
            continue
            
        path_str = str(md_file.resolve())
        is_paper = is_arxiv_paper(md_file)
        
        # 计算该文件在 Chroma 中的 ID
        chroma_id = f"obsidian_{compute_hash(path_str)[:16]}"
        
        if is_paper:
            all_papers.append((path_str, chroma_id))
        else:
            all_notes.append((path_str, chroma_id))

    print(f"📂 物理检查: 论文({len(all_papers)}), 普通笔记({len(all_notes)})")

    # 4. 识别差项 (Missing)
    to_lance = []  # 仅论文
    to_chroma = [] # 所有
    
    # 检查 LanceDB (仅针对论文)
    for p_path, c_id in all_papers:
        if p_path not in lance_sources:
            to_lance.append(p_path)
            
    # 检查 ChromaDB (针对所有文件)
    for p_path, c_id in all_papers + all_notes:
        if c_id not in chroma_ids:
            to_chroma.append((p_path, c_id))
            
    # 5. 识别陈旧项 (Stale)
    stale_lance = []
    for s in lance_sources:
        if s.startswith(str(VAULT_ROOT)) and not os.path.exists(s):
            stale_lance.append(s)
            
    print(f"✨ 待补全: LanceDB({len(to_lance)}), ChromaDB({len(to_chroma)})")
    print(f"🧹 待清理: LanceDB({len(stale_lance)})")
    
    if dry_run:
        print("💡 [Dry Run] 扫描完成，未执行任何修改。")
        return
    
    # 6. 执行同步补全
    # 6.1 LanceDB 批量补全 (V9.0 优化)
    lance_success = 0
    if to_lance:
        if limit > 0 and len(to_lance) > limit:
            print(f"⏩ [Limit Applied] 仅处理前 {limit} 个 LanceDB 任务 (共 {len(to_lance)})")
            to_lance = to_lance[:limit]

        # 将工作列表写入临时文件
        list_file = "/tmp/ingest_work_list.txt"
        with open(list_file, "w") as f:
            f.write("\n".join(to_lance))

        print(f"📦 已准备批量任务列表 ({len(to_lance)} 个文件)，启动 ingest_batch.py...")
        batch_script = Path(__file__).parent / "ingest_batch.py"

        # 一次性调用，持久化模型
        res = subprocess.run([sys.executable, str(batch_script), "--list", list_file], capture_output=False, text=True)
        if res.returncode == 0:
            lance_success = len(to_lance)
        
    # 6.2 ChromaDB 补全
    chroma_success = 0
    to_chroma_final = to_chroma
    if limit > 0 and len(to_chroma) > limit:
        print(f"⏩ [Limit Applied] 仅处理前 {limit} 个 ChromaDB 任务 (共 {len(to_chroma)})")
        to_chroma_final = to_chroma[:limit]

    for p_path, c_id in to_chroma_final:
        print(f"📡 [Chroma] 向量化: {os.path.basename(p_path)}")
        if vectorize_to_chroma(Path(p_path), c_id):
            chroma_success += 1
        time.sleep(0.5) # 频率限制保护

    # 7. 执行陈旧项清理
    for s in stale_lance:
        print(f"🗑️ [Lance] 移除陈旧索引: {os.path.basename(s)}")
        try:
            store.delete_by_source(s)
        except Exception as e: print(f"  ❌ Lance 清理异常: {e}")
            
    print(f"\n✅ 同步对齐完成! Lance(+{lance_success}), Chroma(+{chroma_success}), 清理({len(stale_lance)})")

def main():
    parser = argparse.ArgumentParser(description="VaultSyncMaster - 统一同步工具")
    parser.add_argument("--dry-run", action="store_true", help="只检查不执行")
    parser.add_argument("--limit", type=int, default=0, help="处理任务数量上限 (0 为不限制)")
    parser.add_argument("--skip-chroma", action="store_true", help="跳过 ChromaDB 检查 (用于处理容器故障)")
    args = parser.parse_args()
    
    reconcile(dry_run=args.dry_run, limit=args.limit, skip_chroma=args.skip_chroma)

if __name__ == "__main__":
    main()
