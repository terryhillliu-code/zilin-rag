"""
集成化网页阅读器 (v5.3)
- 遵循“不重复造轮子”原则
- 使用 Jina Reader (r.jina.ai) 作为核心抓取平台
- 仅负责调用集成 API，不编写底层爬虫逻辑
"""
import requests
import sys
from typing import Optional

class WebReader:
    """网页内容读取器（集成版）"""
    
    def __init__(self, use_jina: bool = True):
        self.use_jina = use_jina
        self.jina_prefix = "https://r.jina.ai/"

    def fetch_markdown(self, url: str) -> Optional[str]:
        """
        通过集成平台获取网页的 Markdown 内容
        """
        if self.use_jina:
            return self._fetch_via_jina(url)
        return None

    def _fetch_via_jina(self, url: str) -> Optional[str]:
        """使用 Jina Reader 转换 URL"""
        jina_url = f"{self.jina_prefix}{url}"
        print(f"[WebReader] 正在通过 Jina Reader 请求: {url}", file=sys.stderr)
        
        try:
            # Jina Reader 基础版无需 API Key，但可通过 Header 增强
            headers = {
                "Accept": "text/event-stream" if False else "text/plain"
            }
            resp = requests.get(jina_url, headers=headers, timeout=30)
            
            if resp.status_code == 200:
                content = resp.text
                if len(content) < 100:
                    print(f"[WebReader] 警告：获取内容过短", file=sys.stderr)
                return content
            else:
                print(f"[WebReader] 错误：Jina 返回状态码 {resp.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"[WebReader] 异常: {e}", file=sys.stderr)
        
        return None

# 快捷函数
def get_web_markdown(url: str) -> Optional[str]:
    reader = WebReader()
    return reader.fetch_markdown(url)
