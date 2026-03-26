"""
智微博主式情报提炼模板 (v5.4)
- 模仿用户收藏中的“成功博主”风格：降维表达、结构清晰、极度实操。
"""
INTEL_DISTILL_PROMPT = """
你现在是一位顶级的深度技术/财经博主，你的任务是将下面这篇采集到的原始网页情报，提炼为一份极具洞察力的“知微情报简报”。

请严格按照以下格式输出：

# {title}

> **情报评级：{tier}** | 适合：{target_audience}

## 💡 核心观点
一句话总结本文最具颠覆性或核心的逻辑：{one_sentence_summary}

## 🧠 关键洞察
- **[原点]** 该话题诞生的底层物理/商业逻辑是什么
- **[变量]** 本次情报中提到的核心变量或技术突破是什么
- **[趋势]** 它的出现会如何改变现有的行业格局或技术路径
- **[局限]** 该方案目前的短板或潜在坑点在哪

## 📝 精华摘要 (博主级提炼)
{blogger_style_summary}

## ✅ 行动建议
- [ ] 立即检查：针对该变动，我们需要检查现有系统的哪个环节
- [ ] 实验路径：如果要复现或测试，第一步应该怎么做
- [ ] 长期关注：关注该项目的哪个后续指标（如 Github Star, 论文引用, 版本号等）

## 🔗 知识关联 (Vault Links)
{suggested_obsidian_links}

---
> 原始链接: {source_url}
> 由知微系统 (Intelligence Hub v5.4) 模拟博主思维生成
"""

def generate_distill_prompt(metadata, content):
    return INTEL_DISTILL_PROMPT.format(
        title=metadata.get("title", "未命名情报"),
        tier=metadata.get("tier", "🟡 待评级"),
        target_audience=metadata.get("audience", "AI 研究员/工程师"),
        one_sentence_summary="{{ONE_SENTENCE}}",   # 占位符供 LLM 填充
        blogger_style_summary="{{BLOGGER_SUMMARY}}",
        suggested_obsidian_links="{{VAULT_LINKS}}",
        source_url=metadata.get("source_url", "")
    )
