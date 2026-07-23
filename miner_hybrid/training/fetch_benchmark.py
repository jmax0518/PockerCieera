"""Download Poker44 public training benchmark chunks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://api.poker44.net/api/v1/benchmark"


def _get(url: str, params: dict[str, Any] | None = None, timeout: int = 60) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Benchmark API error: {payload}")
    return payload["data"]


def fetch_status() -> dict[str, Any]:
    return _get(API_BASE)


def fetch_chunks_for_date(
    source_date: str,
    *,
    limit: int = 24,
    split: str | None = None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    cursor = None
    while True:
        params: dict[str, Any] = {"sourceDate": source_date, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if split:
            params["split"] = split
        data = _get(f"{API_BASE}/chunks", params=params, timeout=120)
        page = data.get("chunks") or []
        chunks.extend(page)
        cursor = data.get("nextCursor")
        if not cursor:
            break
        time.sleep(0.05)
    return chunks


def save_release(out_dir: Path, source_date: str, chunks: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"benchmark_{source_date}.json"
    payload = {
        "sourceDate": source_date,
        "chunkCount": len(chunks),
        "chunks": chunks,
    }
    path.write_text(json.dumps(payload))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Poker44 public benchmark")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "benchmark"),
    )
    parser.add_argument("--days", type=int, default=7, help="How many latest dates to pull")
    parser.add_argument("--source-date", default="", help="Single YYYY-MM-DD override")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    status = fetch_status()
    print(
        f"benchmark latest={status.get('latestSourceDate')} "
        f"totalChunks={status.get('totalChunks')} "
        f"release={status.get('releaseVersion')}"
    )

    if args.source_date:
        dates = [args.source_date]
    else:
        releases = _get(f"{API_BASE}/releases", params={"limit": max(1, args.days)})
        dates = [str(item["sourceDate"]) for item in (releases.get("releases") or releases or [])]
        if not dates and status.get("latestSourceDate"):
            dates = [str(status["latestSourceDate"])]
        dates = dates[: max(1, args.days)]

    for source_date in dates:
        print(f"downloading {source_date} ...")
        chunks = fetch_chunks_for_date(source_date, limit=args.limit)
        path = save_release(out_dir, source_date, chunks)
        print(f"  saved {len(chunks)} chunks -> {path}")


if __name__ == "__main__":
    main()
