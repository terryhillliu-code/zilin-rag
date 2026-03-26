"""
知微情报同步引擎 (v5.5)
- 定期运行，同步 config/intel_sources.yaml 中的信源
- 使用 url_ingest --distill 进行深度提炼
- 防止重复同步（基于 URL 记录）
"""
import os
import sys
import yaml
import hashlib
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.url_ingest import url_ingest

# 配置
INTEL_SOURCES_FILE = Path(__file__).parent.parent / "config" / "intel_sources.yaml"
SYNC_STATE_FILE = Path(__file__).parent.parent / "data" / "intel_sync_state.json"

def load_sync_state():
    """加载同步状态（已同步的 URL）"""
    if SYNC_STATE_FILE.exists():
        try:
            return set(json.loads(SYNC_STATE_FILE.read_text(encoding='utf-8')))
        except:
            return set()
    return set()

def save_sync_state(state):
    """保存同步状态"""
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_FILE.write_text(json.dumps(list(state), indent=2), encoding='utf-8')

def intel_sync():
    """执行情报同步"""
    if not INTEL_SOURCES_FILE.exists():
        print(f"❌ 找不到配置文件: {INTEL_SOURCES_FILE}")
        return

    with open(INTEL_SOURCES_FILE, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sources = config.get("sources", [])
    synced_urls = load_sync_state()
    newly_synced = 0

    print(f"🔄 开始同步 {len(sources)} 个情报源...")

    for source in sources:
        name = source.get("name")
        url = source.get("url")
        s_type = source.get("type", "blog")
        
        print(f"--- 检查: [{name}] ({s_type}) ---")
        
        # 简单去重逻辑
        if url in synced_urls:
            print(f"⏭️  已同步过，跳过。")
            continue

        # 如果是 RSS，我们需要解析它 (这里先实现 Blog/Single URL 逻辑，后续可扩展 feedparser)
        if s_type == "blog" or s_type == "web":
            success = url_ingest(url, title=name, distill=True)
            if success:
                synced_urls.add(url)
                newly_synced += 1
        elif s_type == "rss":
            # 基础 RSS 解析逻辑
            try:
                import feedparser
                feed = feedparser.parse(url)
                # 只取最近 3 篇，避免瞬间刷屏
                for entry in feed.entries[:3]:
                    e_url = entry.link
                    if e_url not in synced_urls:
                        print(f"📰 发现新内容: {entry.title}")
                        success = url_ingest(e_url, title=entry.title, distill=True)
                        if success:
                            synced_urls.add(e_url)
                            newly_synced += 1
                    else:
                        print(f"⏭️  内容已存在: {entry.title}")
            except ImportError:
                print(f"⚠️  请安装 feedparser (pip install feedparser) 以支持 RSS 类型")
            except Exception as e:
                print(f"❌ RSS 解析失败 [{name}]: {e}")
        elif s_type == "trend":
            print(f"📡 监听模式: {name} 由 TrendBridge 受理推送。")
            # 这里可以增加一些健康检查逻辑

    save_sync_state(synced_urls)
    print(f"✅ 同步完成！新增情报: {newly_synced}")

if __name__ == "__main__":
    intel_sync()
