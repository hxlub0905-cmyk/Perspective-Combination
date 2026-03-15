"""
record_skip.py — Record a skipped suggestion to the history GitHub Gist.

The suggest.py script reads this history to avoid re-suggesting items
that are still in their 30-day cooldown window.
"""

import os
import json
import datetime
import requests


def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Fusi3-AI-Bot",
    }


def main():
    print("📋 record_skip.py starting...")

    gh_token = os.environ["GITHUB_TOKEN"]
    gist_id = os.environ.get("HISTORY_GIST_ID", "").strip()
    suggestion = json.loads(os.environ.get("SUGGESTION_JSON", "{}"))

    headers = _gh_headers(gh_token)

    # ── Load existing history ─────────────────────────────────────────────────
    history: list = []
    if gist_id:
        r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=10)
        if r.ok:
            raw = r.json().get("files", {}).get("history.json", {}).get("content", "[]")
            try:
                history = json.loads(raw)
            except json.JSONDecodeError:
                history = []

    # ── Append skip record ────────────────────────────────────────────────────
    today = datetime.date.today()
    suggest_again = today + datetime.timedelta(days=30)
    record = {
        "action": "skip",
        "title": suggestion.get("title", ""),
        "category": suggestion.get("category", ""),
        "date": today.isoformat(),
        "suggest_again_after": suggest_again.isoformat(),
    }
    history.append(record)
    history = history[-200:]  # keep last 200 records

    new_content = json.dumps(history, ensure_ascii=False, indent=2)

    # ── Save to gist ──────────────────────────────────────────────────────────
    if gist_id:
        r = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers=headers,
            json={"files": {"history.json": {"content": new_content}}},
            timeout=10,
        )
        r.raise_for_status()
        print(f"✅ Updated history gist {gist_id}")
        print(f"   Skipped: '{record['title']}' — will re-suggest after {suggest_again}")
    else:
        # First run — create the gist and print its ID for the user to save
        r = requests.post(
            "https://api.github.com/gists",
            headers=headers,
            json={
                "description": "Fusi³ AI Bot – Suggestion History",
                "public": False,
                "files": {"history.json": {"content": new_content}},
            },
            timeout=10,
        )
        r.raise_for_status()
        new_gist_id = r.json()["id"]
        print(f"✅ Created new history gist: {new_gist_id}")
        print("⚠️  ACTION REQUIRED: Add this as HISTORY_GIST_ID in GitHub Secrets!")


if __name__ == "__main__":
    main()
