"""告警推送模块

统一管理飞书/钉钉告警，复用 zhiwei-scheduler 的推送队列机制。
"""
import time
from pathlib import Path
from datetime import datetime, timezone
import json


def send_alert(title: str, content: str, level: str = "error", push_targets: list[str] = ["feishu"]):
    """发送告警推送

    Args:
        title: 告警标题
        content: 告警内容
        level: 告警级别 (error, warning, info)
        push_targets: 推送目标列表
    """
    try:
        pending_dir = Path.home() / "zhiwei-scheduler" / "outputs" / "artifacts" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        job_id = f"alert_{level}_{int(time.time())}"

        level_icons = {
            "error": "🚨",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        icon = level_icons.get(level, "📢")

        payload = {
            "job_id": job_id,
            "task": "push_alert",
            "content": f"# {icon} {title}\n\n{content}",
            "push_targets": push_targets,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retries": 0,
            "last_error": None,
            "metadata": {"type": level, "source": "zhiwei-rag"}
        }

        payload_file = pending_dir / f"{job_id}.json"
        with open(payload_file, 'w') as f:
            json.dump(payload, f, indent=2)

        return True
    except Exception as e:
        print(f"告警发送失败: {e}")
        return False


def alert_ingest_failure(filename: str, error_msg: str, source: str = "unknown"):
    """入库失败告警"""
    return send_alert(
        title="入库失败",
        content=f"**文件**: {filename}\n**来源**: {source}\n**错误**: {error_msg}\n\n> 请检查 `zhiwei-rag` 日志或手动处理。",
        level="error"
    )


def alert_index_sync_failure(moved_count: int, error_msg: str):
    """索引同步失败告警"""
    return send_alert(
        title="索引同步失败",
        content=f"**移动文件数**: {moved_count}\n**错误**: {error_msg}\n\n> 文件已移动但索引未更新，可能影响搜索结果。",
        level="warning"
    )


def alert_batch_ingest_partial_failure(success: int, failed: int, error_msg: str):
    """批量入库部分失败告警"""
    if failed == 0:
        return True  # 无失败，不发告警

    return send_alert(
        title="批量入库部分失败",
        content=f"**成功**: {success}\n**失败**: {failed}\n**错误**: {error_msg}\n\n> 失败的文件可能需要手动补录。",
        level="warning" if failed < success else "error"
    )