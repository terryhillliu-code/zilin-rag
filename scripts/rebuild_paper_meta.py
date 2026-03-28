import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

# Paths
DB_PATH = "/Users/liufang/zhiwei-rag/data/arxiv_meta.db"
OUTPUTS_DIR = "/Users/liufang/zhiwei-scheduler/outputs"

def init_db(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            published TEXT,
            url TEXT,
            summary TEXT,
            categories TEXT,
            score REAL,
            ingested_at TEXT
        )
    """)
    conn.commit()

def process_file(conn, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cur = conn.cursor()
    count = 0
    for p in data.get("papers", []):
        arxiv_id = p["url"].split("/")[-1]
        cur.execute("""
            INSERT OR REPLACE INTO papers (arxiv_id, title, authors, published, url, summary, categories, score, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            arxiv_id,
            p["title"],
            ", ".join(p["authors"]),
            p["published"],
            p["url"],
            p["summary"],
            ", ".join(p["categories"]),
            p.get("final_score", 0),
            datetime.now().isoformat()
        ))
        count += 1
    conn.commit()
    return count

def rebuild():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    
    total = 0
    raw_files = list(Path(OUTPUTS_DIR).glob("raw_arxiv_*.json"))
    print(f"🚀 Found {len(raw_files)} raw arxiv JSON files.")
    
    for rf in raw_files:
        print(f"   Processing: {rf.name}...")
        try:
            c = process_file(conn, rf)
            total += c
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
    conn.close()
    print(f"\n📊 Rebuild Complete! Total papers in metadata DB: {total}")

if __name__ == "__main__":
    rebuild()
