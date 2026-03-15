"""line_utils.py — LINE Messaging API push message helper."""

import os
import requests


def send_line_message(text: str) -> bool:
    """Push a text message to the owner via LINE Messaging API.

    Required env vars:
        LINE_CHANNEL_TOKEN  — Channel Access Token
        LINE_USER_ID        — Your LINE User ID (Uxxxxxxxxxx)

    Returns True on success, False on failure (so callers can fall back gracefully).
    """
    token = os.environ.get("LINE_CHANNEL_TOKEN", "")
    user_id = os.environ.get("LINE_USER_ID", "")

    if not token or not user_id:
        print("  ⚠️  LINE_CHANNEL_TOKEN or LINE_USER_ID not set — skipping LINE notify")
        return False

    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": text}],
    }
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json=payload,
        timeout=10,
    )
    if r.ok:
        print("  ✅ LINE message sent")
        return True
    else:
        print(f"  ⚠️  LINE push failed: {r.status_code} {r.text}")
        return False


def format_suggestion_message(suggestion: dict) -> str:
    """Format a suggestion as a LINE text message."""
    cat_emoji = {"UX": "🎨", "Bug": "🐛", "Feature": "✨", "Refactor": "🔧"}.get(
        suggestion.get("category", ""), "💡"
    )
    complexity = suggestion.get("estimated_complexity", "")
    core_warn = "\n⚠️ 核心演算法異動 — 需手動審核" if suggestion.get("requires_core_review") else ""
    files = ", ".join(suggestion.get("files_affected", []))

    return (
        f"🤖 Fusi³ AI 今日建議\n"
        f"{'─' * 20}\n"
        f"{cat_emoji} {suggestion.get('title', '')}\n"
        f"類別：{suggestion.get('category', '')}  複雜度：{complexity}\n\n"
        f"{suggestion.get('description', '')}\n"
        f"{core_warn}\n\n"
        f"📁 {files}\n\n"
        f"➡️ 請至 Gmail 點選 OK / 修改方向 / Skip"
    ).strip()


def format_pr_message(suggestion: dict, pr_url: str) -> str:
    """Format a PR completion notice as LINE text."""
    return (
        f"✅ Fusi³ AI 實作完成！\n"
        f"{'─' * 20}\n"
        f"{suggestion.get('title', '')}\n\n"
        f"🔗 PR: {pr_url}\n\n"
        f"請晚上 review 後 merge。"
    )


def format_error_message(suggestion: dict, run_url: str) -> str:
    """Format an error notice as LINE text."""
    return (
        f"❌ Fusi³ AI 實作失敗\n"
        f"{'─' * 20}\n"
        f"{suggestion.get('title', '')}\n\n"
        f"🔍 查看 log：{run_url}"
    )
