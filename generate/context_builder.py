"""
上下文组装器
- 将检索结果组装为结构化 Prompt
- Token 预算控制
- 来源标注
"""
import os
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class ContextConfig:
    """上下文配置"""
    max_tokens: int = 4000           # 上下文 token 预算
    chars_per_token: float = 1.5     # 中文字符/token 估算
    include_source: bool = True      # 是否包含来源标注
    include_score: bool = False      # 是否包含分数
    separator: str = "\n\n---\n\n"   # 片段分隔符


class ContextBuilder:
    """上下文组装器"""
    
    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self.templates_dir = Path(__file__).parent / "prompt_templates"
    
    def build(
        self,
        query: str,
        results: list,  # list[RerankResult] 或 list[RetrievalResult]
        template_name: str = "qa",
        extra_context: Optional[str] = None
    ) -> str:
        """
        构建完整的 Prompt
        
        Args:
            query: 用户查询
            results: 检索结果
            template_name: 模板名称 (qa/brief/...)
            extra_context: 额外上下文（如图谱全局信息）
            
        Returns:
            组装好的 Prompt
        """
        # 加载模板
        template = self._load_template(template_name)
        
        # 组装检索上下文
        retrieval_context = self._build_retrieval_context(results)
        
        # 计算已用 token
        used_tokens = self._estimate_tokens(template) + self._estimate_tokens(query)
        if extra_context:
            used_tokens += self._estimate_tokens(extra_context)
        
        # 裁剪检索上下文以适应预算
        available_tokens = self.config.max_tokens - used_tokens - 100  # 留 100 token 余量
        retrieval_context = self._truncate_to_budget(retrieval_context, available_tokens)
        
        # 填充模板
        prompt = template.format(
            query=query,
            context=retrieval_context,
            extra_context=extra_context or "",
            source_count=len(results)
        )
        
        return prompt
    
    def build_context_only(self, results: list) -> str:
        """仅构建检索上下文（不含模板）"""
        context = self._build_retrieval_context(results)
        return self._truncate_to_budget(context, self.config.max_tokens)
    
    def _build_retrieval_context(self, results: list) -> str:
        """构建检索上下文"""
        if not results:
            return "(无相关检索结果)"
        
        parts = []
        for i, r in enumerate(results):
            part = self._format_result(r, index=i + 1)
            parts.append(part)
        
        return self.config.separator.join(parts)
    
    def _format_result(self, result, index: int) -> str:
        """格式化单个检索结果"""
        lines = []

        # 来源标注
        if self.config.include_source:
            source = getattr(result, 'source', 'unknown')
            # 简化路径显示
            if '/' in source:
                source = source.split('/')[-1]

            # MM-005: 增强引用信息
            metadata = getattr(result, 'metadata', {})
            chunk_type = metadata.get('chunk_type', 'text')
            page = metadata.get('page', 0)

            # 构建来源标签
            type_label = {"text": "文字", "figure": "图表", "table": "表格", "frame": "视频帧"}.get(chunk_type, "内容")
            if page > 0:
                source_label = f"[来源 {index}: {source} 第{page}页 {type_label}]"
            elif chunk_type != "text":
                source_label = f"[来源 {index}: {source} {type_label}]"
            else:
                source_label = f"[来源 {index}: {source}]"

            lines.append(source_label)

        # 分数（可选）
        if self.config.include_score:
            if hasattr(result, 'rerank_score'):
                lines.append(f"相关度: {result.rerank_score:.2f}")
            elif hasattr(result, 'score'):
                lines.append(f"相关度: {result.score:.2f}")

        # 内容
        text = getattr(result, 'text', str(result))
        lines.append(text)

        return '\n'.join(lines)
    
    def _load_template(self, name: str) -> str:
        """加载 Prompt 模板"""
        template_path = self.templates_dir / f"{name}.txt"
        
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
        
        # 默认模板
        return self._get_default_template(name)
    
    def _get_default_template(self, name: str) -> str:
        """获取默认模板"""
        if name == "qa":
            return """你是一个知识助手。请根据以下检索到的参考资料回答用户的问题。

## 参考资料（共 {source_count} 条）

{context}

{extra_context}

## 用户问题

{query}

## 回答要求

1. 基于参考资料回答，不要编造信息
2. 如果参考资料不足以回答，请明确说明
3. 适当引用来源编号，如 [来源 1]
4. 使用清晰的结构组织答案

请回答："""
        
        elif name == "brief":
            return """你是一个信息分析师。请根据以下资料生成简报。

## 参考资料（共 {source_count} 条）

{context}

{extra_context}

## 简报主题

{query}

## 要求

1. 提取关键信息，去除冗余
2. 按重要性排序
3. 标注信息来源
4. 控制篇幅在 500 字以内

请生成简报："""
        
        else:
            # 通用模板
            return """## 参考资料

{context}

{extra_context}

## 任务

{query}

请基于以上资料完成任务："""
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数"""
        if not text:
            return 0
        return int(len(text) / self.config.chars_per_token)
    
    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """裁剪文本以适应 token 预算"""
        max_chars = int(max_tokens * self.config.chars_per_token)
        
        if len(text) <= max_chars:
            return text
        
        # 按分隔符切分，保留前面的部分
        parts = text.split(self.config.separator)
        result_parts = []
        current_len = 0
        
        for part in parts:
            part_len = len(part) + len(self.config.separator)
            if current_len + part_len > max_chars:
                break
            result_parts.append(part)
            current_len += part_len
        
        if result_parts:
            truncated = self.config.separator.join(result_parts)
            truncated += f"\n\n(已截断，共 {len(parts)} 条资料，显示前 {len(result_parts)} 条)"
            return truncated
        
        # 如果第一个部分就超长，强制截断
        return text[:max_chars] + "...(已截断)"


# 快捷函数
def build_context(
    query: str,
    results: list,
    template: str = "qa",
    max_tokens: int = 4000
) -> str:
    """快捷上下文构建"""
    config = ContextConfig(max_tokens=max_tokens)
    builder = ContextBuilder(config)
    return builder.build(query, results, template_name=template)
