#!/usr/bin/env python3
"""
zhiwei-rag 桥接脚本
供 zhiwei-scheduler 通过子进程调用

用法:
    python bridge.py retrieve "查询文本" --top-k 5
    python bridge.py context "查询文本" --top-k 5
    python bridge.py prompt "查询文本" --template qa --top-k 5
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_retrieve(args):
    """执行检索，输出 JSON"""
    from api import retrieve
    
    results = retrieve(args.query, top_k=args.top_k)
    
    output = []
    for r in results:
        item = {
            'text': r.text,
            'raw_text': r.raw_text,
            'source': r.source,
            'track': r.track,
        }
        if hasattr(r, 'rerank_score'):
            item['score'] = r.rerank_score
        else:
            item['score'] = r.score
        output.append(item)
    
    print(json.dumps(output, ensure_ascii=False, indent=2))


def cmd_context(args):
    """获取检索上下文，输出纯文本"""
    from api import get_context
    
    context = get_context(args.query, top_k=args.top_k)
    print(context)


def cmd_prompt(args):
    """构建完整 Prompt，输出纯文本"""
    from api import retrieve_and_build_prompt
    
    prompt = retrieve_and_build_prompt(
        args.query,
        top_k=args.top_k,
        template=args.template
    )
    print(prompt)


def main():
    parser = argparse.ArgumentParser(description='zhiwei-rag bridge')
    subparsers = parser.add_subparsers(dest='command')
    
    # retrieve
    p1 = subparsers.add_parser('retrieve')
    p1.add_argument('query')
    p1.add_argument('--top-k', type=int, default=5)
    p1.set_defaults(func=cmd_retrieve)
    
    # context
    p2 = subparsers.add_parser('context')
    p2.add_argument('query')
    p2.add_argument('--top-k', type=int, default=5)
    p2.set_defaults(func=cmd_context)
    
    # prompt
    p3 = subparsers.add_parser('prompt')
    p3.add_argument('query')
    p3.add_argument('--top-k', type=int, default=5)
    p3.add_argument('--template', default='qa')
    p3.set_defaults(func=cmd_prompt)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
