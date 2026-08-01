import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "llmkit" / "traces" / "llmkit_traces.db"

REQUIRED_FIELDS = ["model", "prompt", "input_tokens", "output_tokens", "cost_usd", "latency_ms"]


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("SELECT * FROM traces ORDER BY id").fetchall()
    total = len(rows)
    success_rows = [r for r in rows if r["status"] == "success"]
    error_rows = [r for r in rows if r["status"] == "error"]

    print(f"Total records: {total}")
    print(f"  success: {len(success_rows)}")
    print(f"  error:   {len(error_rows)}")
    print(f"  retried: {sum(1 for r in rows if r['retried'])}")

    incomplete = []
    for r in success_rows:
        missing = [f for f in REQUIRED_FIELDS if r[f] is None]
        try:
            output = json.loads(r["response"])["output"] if r["response"] else None
        except (TypeError, ValueError, KeyError):
            output = None
        if not output:
            missing.append("output")
        if missing:
            incomplete.append((r["id"], missing))

    print(f"\nComplete success records (has {', '.join(REQUIRED_FIELDS)}, output): "
          f"{len(success_rows) - len(incomplete)}/{len(success_rows)}")

    if incomplete:
        print("Incomplete records:")
        for row_id, missing in incomplete:
            print(f"  id={row_id} missing={missing}")

    if rows:
        print("\nSample record:")
        sample = dict(rows[0])
        for key, value in sample.items():
            if key == "response" and value:
                value = value[:200] + ("..." if len(value) > 200 else "")
            if key == "prompt" and value:
                value = value[:200] + ("..." if len(value) > 200 else "")
            print(f"  {key}: {value}")

    conn.close()


if __name__ == "__main__":
    main()
