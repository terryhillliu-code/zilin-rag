#!/usr/bin/env python3
"""
Zhiwei TrendBridge (v5.6)
- 接收来自 TrendRadar 或其他平台的热点推送 (Webhook)
- 执行 AI 降噪，过滤非技术/娱乐类热点
- 调用 url_ingest --distill 实现自动化闭环入库
"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.url_ingest import url_ingest

# 配置
BRIDGE_LOG_DIR = Path(__file__).parent.parent / "data" / "trend_bridge"
BRIDGE_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BRIDGE_LOG_DIR / "bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrendBridge")

app = FastAPI(title="Zhiwei TrendBridge")

class TrendItem(BaseModel):
    title: str
    url: str
    source: str = "unknown"
    hot_score: Optional[float] = 0.0
    platform: str = "web"
    tags: List[str] = []

def process_trend_task(item: TrendItem):
    """后台处理热点任务"""
    logger.info(f"🚀 处理热点: {item.title} ({item.url})")
    
    # 1. 简单的技术关键词过滤 (降噪)
    tech_keywords = ["AI", "LLM", "GPT", "DeepSeek", "模型", "架构", "芯片", "GPU", "开源", "智能", "机器人", "Agent"]
    is_tech = any(kw.lower() in item.title.lower() for kw in tech_keywords)
    
    if not is_tech:
        logger.info(f"⏭️  非技术相关，跳过: {item.title}")
        return

    # 2. 调用 url_ingest 进行深度提炼
    try:
        # TODO: 后续可接入 LLM 再次判定价值
        success = url_ingest(item.url, title=item.title, distill=True)
        if success:
            logger.info(f"✅ 热点已入库: {item.title}")
            # 记录到历史
            history_file = BRIDGE_LOG_DIR / "history.jsonl"
            with open(history_file, "a", encoding='utf-8') as f:
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "title": item.title,
                    "url": item.url,
                    "status": "success"
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            logger.error(f"❌ 入库失败: {item.title}")
    except Exception as e:
        logger.error(f"⚠️ 处理异常: {e}")

@app.post("/webhook/trend")
async def receive_trend(item: TrendItem, background_tasks: BackgroundTasks):
    """接收热点推送接口"""
    if not item.url or not item.title:
        raise HTTPException(status_code=400, detail="Missing url or title")
    
    background_tasks.add_task(process_trend_task, item)
    return {"status": "queued", "title": item.title}

@app.get("/status")
async def get_status():
    """查看运行状态"""
    return {
        "service": "TrendBridge",
        "updated": datetime.now().isoformat(),
        "log_path": str(BRIDGE_LOG_DIR)
    }

if __name__ == "__main__":
    import uvicorn
    # 获取环境变量中的端口，默认 18790 (避开 Docker 18789)
    port = int(os.environ.get("TREND_BRIDGE_PORT", 18790))
    uvicorn.run(app, host="0.0.0.0", port=port)
