# zhiwei-rag 索引机制说明

> 为AI Agent提供统一的索引操作指南

## 快速参考

| 场景 | 命令 |
|------|------|
| **单个文件增量索引** | `~/zhiwei-shared-venv/bin/python ~/zhiwei-rag/scripts/ingest_incremental.py --file /path/to/file.md` |
| **批量云端索引（推荐）** | `~/zhiwei-shared-venv/bin/python ~/zhiwei-rag/scripts/ingest_cloud_fast.py --list /tmp/file_list.txt` |
| **清理陈旧索引** | `~/zhiwei-shared-venv/bin/python ~/zhiwei-rag/scripts/clean_stale_lance.py` |
| **清理重复索引** | `~/zhiwei-shared-venv/bin/python ~/zhiwei-rag/scripts/clean_duplicates_lance.py` |
| **全量同步修复** | `~/zhiwei-shared-venv/bin/python ~/zhiwei-rag/scripts/reconcile_obsidian.py` |

## 新论文处理流程

### 自动流程（已有）
arxiv-paper-analyzer → sync_to_rag.py → ingest_incremental.py

### 手动触发（更快）
```bash
# 生成待索引列表
~/zhiwei-shared-venv/bin/python -c "
from pathlib import Path
vault = Path '~/Documents/ZhiweiVault'
missing = [str(f) for f in vault.glob('**/*.md') 
           if 'Inbox' not in str(f) and 'Archive' not in str(f)]
with open('/tmp/missing.txt', 'w') as f:
    f.write('\\n'.join(missing))
"

# 云端批量索引（推荐）
~/zhiwei-shared-venv/bin/python ~/zhiwei-rag/scripts/ingest_cloud_fast.py --list /tmp/missing.txt
```

## 注意事项

1. **inbox_triage.py 移动文件后需要手动索引**
   - 移动文件不自动更新索引
   - 需运行 `reconcile_obsidian.py` 或手动索引新位置

2. **DKI排除目录**
   - Inbox, Archive, attachments, 72_视频笔记 等目录不索引
   - 见 config.yaml 的 `dki.skip_dirs`

3. **索引完整性检查**
   ```bash
   ~/zhiwei-shared-venv/bin/python -c "
   import lancedb
   db = lancedb.connect '~/zhiwei-rag/data/lance_db'
   tbl = db.open_table('documents')
   print(f'索引总数: {tbl.count_rows()}')
   "
   ```

## 常见问题

**Q: 新论文入库后搜不到？**
A: 运行 `sync_to_rag.py --batch --limit 100` 或用云端脚本补全

**Q: 发现重复索引？**
A: 运行 `clean_duplicates_lance.py`

**Q: 文件移动后索引失效？**
A: 运行 `reconcile_obsidian.py` 清理陈旧+补全缺失

**Q: 索引速度太慢？**
A: 使用 `ingest_cloud_fast.py`（云端API，提速200倍）