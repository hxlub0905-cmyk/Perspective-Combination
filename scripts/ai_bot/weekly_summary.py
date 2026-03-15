"""
weekly_summary.py — Generate and send a weekly development summary for Fusi³.

Groups commits into:
  - AI Bot suggestions (branches starting with ai/)
  - Manual / vibe coding (all other commits to main)

Fetches merged PR info from GitHub API to get proper titles and descriptions.
Sends summary via Gmail + LINE.
"""

import os
import json
import datetime
import subprocess
import requests
import anthropic
from email_utils import send_email


# ─── Git helpers ──────────────────────────────────────────────────────────────

def get_commits_since(days: int = 7) -> list[dict]:
    """Return commits merged to main in the last N days."""
    since = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    result = subprocess.run(
        ["git", "log", f"--since={since}", "--merges",
         "--pretty=format:%H|%s|%ae|%ad", "--date=short", "origin/main"],
        capture_output=True, text=True
    )
    commits = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "hash": parts[0][:8],
                "subject": parts[1],
                "author": parts[2],
                "date": parts[3],
            })
    return commits


def get_direct_commits(days: int = 7) -> list[dict]:
    """Return direct (non-merge) commits to main in the last N days."""
    since = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    result = subprocess.run(
        ["git", "log", f"--since={since}", "--no-merges",
         "--pretty=format:%H|%s|%ae|%ad", "--date=short", "origin/main"],
        capture_output=True, text=True
    )
    commits = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "hash": parts[0][:8],
                "subject": parts[1],
                "author": parts[2],
                "date": parts[3],
            })
    return commits


# ─── GitHub API ───────────────────────────────────────────────────────────────

def get_merged_prs(gh_token: str, repo: str, days: int = 7) -> list[dict]:
    """Fetch PRs merged in the last N days from GitHub API."""
    since = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {
        "Authorization": f"token {gh_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    r = requests.get(
        f"https://api.github.com/repos/{repo}/pulls",
        headers=headers,
        params={"state": "closed", "base": "main", "per_page": 50, "sort": "updated", "direction": "desc"},
        timeout=15,
    )
    if not r.ok:
        print(f"⚠️  GitHub API error: {r.status_code}")
        return []

    prs = []
    for pr in r.json():
        merged_at = pr.get("merged_at")
        if not merged_at:
            continue
        if merged_at < since:
            continue
        branch = pr.get("head", {}).get("ref", "")
        is_ai_bot = branch.startswith("ai/")
        prs.append({
            "number": pr["number"],
            "title": pr["title"],
            "url": pr["html_url"],
            "branch": branch,
            "merged_at": merged_at[:10],
            "is_ai_bot": is_ai_bot,
            "body": (pr.get("body") or "")[:300],
        })
    return prs


def get_api_usage(gh_token: str, repo: str) -> dict:
    """Try to estimate API cost from Anthropic usage (not available via API, use placeholder)."""
    # Anthropic doesn't provide a cost API — we estimate based on known workflow counts
    return {}


# ─── Claude analysis ──────────────────────────────────────────────────────────

def generate_summary(prs: list[dict], direct_commits: list[dict]) -> str:
    """Use Claude to produce a human-readable Chinese summary."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    ai_prs = [p for p in prs if p["is_ai_bot"]]
    manual_prs = [p for p in prs if not p["is_ai_bot"]]

    def pr_block(items):
        if not items:
            return "（本週無）"
        return "\n".join(
            f"- PR #{p['number']} [{p['merged_at']}] {p['title']}\n  {p['body'][:150]}"
            for p in items
        )

    def commit_block(items):
        if not items:
            return "（本週無）"
        return "\n".join(f"- [{c['date']}] {c['subject']} ({c['author']})" for c in items)

    prompt = f"""你是 Fusi³ 專案的開發助理。請根據以下本週開發紀錄，產生一份給開發者的繁體中文週報摘要。

## AI Bot 自動合併的 PR（suggestions → implemented）
{pr_block(ai_prs)}

## 手動 / Vibe Coding 合併的 PR
{pr_block(manual_prs)}

## 直接 commit 到 main（無 PR）
{commit_block(direct_commits)}

請產生週報，格式如下：
1. 本週亮點（2-3 句話總結最重要的改動）
2. AI Bot 做了什麼（每個 PR 一行說明，重點是「對使用者有什麼影響」）
3. 手動開發 / Vibe Coding（每個 PR/commit 一行說明）
4. 下週建議關注的方向（根據本週進度給 1-2 個建議）

語氣輕鬆，不要太正式，像是同事之間的週報。"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ─── Email builder ────────────────────────────────────────────────────────────

def build_weekly_email(
    summary_text: str,
    ai_prs: list[dict],
    manual_prs: list[dict],
    week_str: str,
) -> tuple[str, str]:
    """Build weekly summary email."""

    def pr_cards(prs: list[dict], color: str) -> str:
        if not prs:
            return '<p style="color:#aaa;font-size:13px;">本週無</p>'
        cards = ""
        for p in prs:
            cards += f"""
            <div style="border:1px solid #e5e5e5;border-radius:8px;padding:10px 14px;margin-bottom:8px;">
              <a href="{p['url']}" style="color:{color};font-weight:600;text-decoration:none;">
                PR #{p['number']} — {p['title']}
              </a>
              <span style="color:#aaa;font-size:11px;margin-left:8px;">{p['merged_at']}</span>
            </div>"""
        return cards

    summary_html = summary_text.replace("\n", "<br>")
    ai_cards = pr_cards(ai_prs, "#FF6B35")
    manual_cards = pr_cards(manual_prs, "#3b82f6")

    subject = f"[Fusi³] 📊 本週開發摘要 {week_str}"
    body = f"""<html><body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:600px;margin:0 auto;padding:20px;color:#1a1a1a;">

<div style="background:#1e293b;color:white;padding:16px 20px;border-radius:10px 10px 0 0;">
  <h2 style="margin:0;font-size:18px;">📊 Fusi³ 本週開發摘要</h2>
  <p style="margin:4px 0 0;opacity:.7;font-size:13px;">{week_str}</p>
</div>

<div style="border:1px solid #e5e5e5;border-top:none;border-radius:0 0 10px 10px;padding:24px;">

  <h4 style="color:#444;margin:0 0 10px;">🤖 AI 分析</h4>
  <div style="background:#f9fafb;border-radius:8px;padding:14px 16px;line-height:1.8;font-size:14px;color:#333;margin-bottom:20px;">
    {summary_html}
  </div>

  <h4 style="color:#FF6B35;margin:0 0 10px;">🤖 AI Bot 合併的 PR</h4>
  {ai_cards}

  <h4 style="color:#3b82f6;margin:20px 0 10px;">👨‍💻 手動開發 / Vibe Coding</h4>
  {manual_cards}

  <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
  <p style="color:#bbb;font-size:11px;">Fusi³ AI Bot — 每週一自動產生</p>
</div>
</body></html>"""

    return subject, body


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("📊 weekly_summary.py starting...")

    gh_token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["REPO_FULL_NAME"]

    today = datetime.date.today()
    week_start = today - datetime.timedelta(days=today.weekday() + 7)
    week_end = week_start + datetime.timedelta(days=6)
    week_str = f"{week_start.strftime('%m/%d')} – {week_end.strftime('%m/%d')}"

    print(f"📅 Week: {week_str}")

    # Collect data
    prs = get_merged_prs(gh_token, repo, days=7)
    direct_commits = get_direct_commits(days=7)
    ai_prs = [p for p in prs if p["is_ai_bot"]]
    manual_prs = [p for p in prs if not p["is_ai_bot"]]

    print(f"  AI PRs: {len(ai_prs)}, Manual PRs: {len(manual_prs)}, Direct commits: {len(direct_commits)}")

    if not prs and not direct_commits:
        print("  No activity this week — skipping summary.")
        return

    # Generate summary
    summary_text = generate_summary(prs, direct_commits)
    print("✅ Summary generated")

    # Send email
    subject, html_body = build_weekly_email(summary_text, ai_prs, manual_prs, week_str)
    send_email(
        to=os.environ["NOTIFY_EMAIL"],
        subject=subject,
        html_body=html_body,
        gmail_user=os.environ["GMAIL_USER"],
        gmail_password=os.environ["GMAIL_APP_PASSWORD"],
    )
    print(f"📧 Weekly summary email sent")

    # LINE notification (brief)
    from line_utils import send_line_message
    line_text = (
        f"📊 Fusi³ 本週摘要 {week_str}\n{'─'*20}\n"
        f"AI Bot PR：{len(ai_prs)} 個\n"
        f"手動開發：{len(manual_prs)} 個 PR + {len(direct_commits)} 個 commit\n\n"
        f"詳細內容請查看 Gmail"
    )
    send_line_message(line_text)
    print("✅ Done!")


if __name__ == "__main__":
    main()
