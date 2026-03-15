"""
suggest.py — Daily AI suggestion generator for Fusi³ (3-choice version)

Flow:
1. Read project source files
2. Load skip/reject history from GitHub Gist
3. Call Claude to generate 3 diverse suggestions
4. Store all 3 in a Gist (gist ID = token)
5. Send email with 3 cards — each with [選這個] [客製化] buttons
"""

import os
import json
import datetime
import requests
import anthropic
from pathlib import Path
from email_utils import send_email


# ─── Project reader ───────────────────────────────────────────────────────────

def read_project_context() -> dict:
    root = Path(".")
    context: dict[str, str] = {}

    small_files = [
        "PROJECT_SUMMARY.md",
        "requirements.txt",
        "perscomb/core/ebeam_snr.py",
        "perscomb/core/roi_set.py",
        "perscomb/ui/design_tokens.py",
    ]
    for f in small_files:
        p = root / f
        if p.exists():
            content = p.read_text(encoding="utf-8")
            if len(content) > 12_000:
                content = content[:12_000] + "\n\n... [truncated]"
            context[f] = content

    pc_path = root / "perscomb/core/perspective_combine.py"
    if pc_path.exists():
        lines = pc_path.read_text(encoding="utf-8").splitlines()
        preview = "\n".join(lines[:600])
        sigs = [l.rstrip() for l in lines if l.lstrip().startswith(("def ", "class "))]
        context["perscomb/core/perspective_combine.py"] = (
            preview + "\n\n# --- remaining signatures ---\n" + "\n".join(sigs[30:])
        )

    dlg_path = root / "perscomb/ui/dialog.py"
    if dlg_path.exists():
        lines = dlg_path.read_text(encoding="utf-8").splitlines()
        header = "\n".join(lines[:250])
        methods = [l.rstrip() for l in lines if l.lstrip().startswith(("def ", "class "))]
        context["perscomb/ui/dialog.py [structure]"] = (
            f"# Total lines: {len(lines)}\n\n"
            + header
            + "\n\n# All methods/classes:\n"
            + "\n".join(methods)
        )

    return context


# ─── History ──────────────────────────────────────────────────────────────────

def load_history(gh_token: str, gist_id: str) -> list:
    if not gist_id:
        return []
    try:
        r = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers=_gh_headers(gh_token), timeout=10
        )
        if r.ok:
            raw = r.json().get("files", {}).get("history.json", {}).get("content", "[]")
            return json.loads(raw)
    except Exception as e:
        print(f"⚠️  Could not load history: {e}")
    return []


def get_blocked_titles(history: list) -> tuple[list[str], list[str]]:
    """Return (skip_cooldown_titles, permanent_reject_lines)."""
    today = datetime.date.today()
    skip_list, reject_list = [], []
    for item in history:
        action = item.get("action")
        if action == "skip":
            try:
                cutoff = datetime.date.fromisoformat(item.get("suggest_again_after", "")[:10])
                if today < cutoff:
                    skip_list.append(item["title"])
            except (ValueError, TypeError):
                pass
        elif action == "reject_permanent":
            reject_list.append(f"- {item['title']}（原因：{item.get('reason', '不適合')}）")
    return skip_list, reject_list


# ─── Claude: generate 3 suggestions ──────────────────────────────────────────

def generate_suggestions(context: dict, skip_titles: list[str], reject_lines: list[str]) -> list[dict]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    files_block = ""
    for name, content in context.items():
        files_block += f"\n\n### {name}\n```\n{content}\n```"

    skip_block = ""
    if skip_titles:
        skip_block = "\n\n暫時略過（本月內請勿重複提）：\n" + "\n".join(f"- {t}" for t in skip_titles)

    reject_block = ""
    if reject_lines:
        reject_block = "\n\n永久拒絕方向（絕對不要提類似的）：\n" + "\n".join(reject_lines)

    prompt = f"""你是 Fusi³ 專案的 AI 開發助理。Fusi³ 是給半導體工程師用的 SEM 影像融合分析桌面工具（PySide6）。

請分析以下程式碼，提出今日 **3 個不同方向** 的改善建議供使用者選擇。
{files_block}
{skip_block}
{reject_block}

要求：
- 3 個建議方向各異（例如：一個 UX、一個 Bug、一個 Feature）
- 每個建議具體且可在 1-3 小時內實作
- 若涉及核心演算法（ebeam_snr.py、perspective_combine.py、roi_set.py），requires_core_review = true

只輸出 JSON 陣列，不要其他文字：
[
  {{
    "title": "簡短標題（中英混合，15字內）",
    "category": "UX | Bug | Feature | Refactor",
    "description": "詳細說明（繁體中文，80字內）",
    "motivation": "Why this matters (English, 50 words max)",
    "files_affected": ["perscomb/ui/dialog.py"],
    "estimated_complexity": "small | medium | large",
    "requires_core_review": false,
    "core_reason": "",
    "implementation_hint": "具體說明要改哪個 function，怎麼改（60字內）"
  }},
  {{ ... }},
  {{ ... }}
]"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())[:3]


# ─── Gist storage ─────────────────────────────────────────────────────────────

def store_suggestions_gist(suggestions: list[dict], gh_token: str) -> str:
    headers = _gh_headers(gh_token)
    titles = " / ".join(s["title"] for s in suggestions)
    payload = {
        "description": f"Fusi³ AI Suggestions – {titles}",
        "public": False,
        "files": {
            "suggestions.json": {
                "content": json.dumps(suggestions, ensure_ascii=False, indent=2)
            }
        },
    }
    r = requests.post("https://api.github.com/gists", headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["id"]


# ─── Email ────────────────────────────────────────────────────────────────────

CAT_EMOJI = {"UX": "🎨", "Bug": "🐛", "Feature": "✨", "Refactor": "🔧"}
COMPLEXITY_COLOR = {"small": "#22c55e", "medium": "#f59e0b", "large": "#ef4444"}
COMPLEXITY_LABEL = {"small": "小改動", "medium": "中等", "large": "大改動"}


def _card(s: dict, idx: int, worker_url: str, gist_id: str) -> str:
    emoji = CAT_EMOJI.get(s["category"], "💡")
    color = COMPLEXITY_COLOR.get(s["estimated_complexity"], "#6b7280")
    label = COMPLEXITY_LABEL.get(s["estimated_complexity"], s["estimated_complexity"])
    files_html = "".join(
        f'<code style="display:block;background:#f4f4f4;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px 0;">{f}</code>'
        for f in s["files_affected"]
    )
    core_badge = (
        '<span style="background:#fff8e1;color:#b45309;font-size:10px;'
        'padding:2px 6px;border-radius:4px;margin-left:6px;border:1px solid #f0c040;">⚠️ 核心異動</span>'
        if s.get("requires_core_review") else ""
    )
    approve_url = f"{worker_url}/approve?token={gist_id}&choice={idx}"
    feedback_url = f"{worker_url}/feedback?token={gist_id}&choice={idx}"

    return f"""
<div style="border:1px solid #e5e7eb;border-radius:12px;padding:18px;margin-bottom:14px;background:#fff;">
  <div style="margin-bottom:8px;">
    <span style="background:#FF6B35;color:white;font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600;">{s['category']}</span>
    <span style="background:#f0f0f0;color:{color};font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600;margin-left:4px;">{label}</span>
    {core_badge}
  </div>
  <h3 style="margin:0 0 6px;font-size:16px;color:#111;">{emoji} {s['title']}</h3>
  <p style="color:#444;font-size:13px;line-height:1.6;margin:0 0 6px;">{s['description']}</p>
  <p style="color:#999;font-size:12px;font-style:italic;line-height:1.5;margin:0 0 8px;">{s['motivation']}</p>
  <div style="margin-bottom:10px;">{files_html}</div>
  <div style="display:flex;gap:8px;">
    <a href="{approve_url}" style="background:#22c55e;color:white;padding:9px 18px;border-radius:8px;text-decoration:none;font-weight:700;font-size:13px;">✅ 選這個</a>
    <a href="{feedback_url}" style="background:#3b82f6;color:white;padding:9px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:13px;">✏️ 客製化</a>
  </div>
</div>"""


def build_email(suggestions: list[dict], worker_url: str, gist_id: str, skip_url: str) -> tuple[str, str]:
    today = datetime.date.today().strftime("%Y年%m月%d日")
    cards = "".join(_card(s, i, worker_url, gist_id) for i, s in enumerate(suggestions))
    date_short = datetime.date.today().strftime("%m/%d")

    subject = f"[Fusi³ AI] 今日 3 個開發方向 {date_short}"
    body = f"""<html><body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:620px;margin:0 auto;padding:20px;color:#1a1a1a;background:#f9fafb;">

<div style="background:#FF6B35;color:white;padding:16px 20px;border-radius:10px 10px 0 0;">
  <h2 style="margin:0;font-size:18px;">🤖 Fusi³ AI 今日開發方向</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px;">{today} — 選一個方向，或客製化後讓 AI 實作</p>
</div>

<div style="border:1px solid #e5e5e5;border-top:none;border-radius:0 0 10px 10px;padding:20px;background:#f9fafb;">
  {cards}
  <div style="text-align:center;margin-top:4px;">
    <a href="{skip_url}" style="color:#bbb;font-size:12px;text-decoration:none;">⏭️ 今天都不做，Skip 全部（一個月後再提）</a>
  </div>
</div>

</body></html>"""

    return subject, body


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Fusi3-AI-Bot",
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🤖 Fusi³ AI suggest.py starting...")

    gh_token = os.environ["GITHUB_TOKEN"]
    worker_url = os.environ["WORKER_URL"].rstrip("/")
    history_gist_id = os.environ.get("HISTORY_GIST_ID", "")

    context = read_project_context()
    print(f"📁 Read {len(context)} context sections")

    history = load_history(gh_token, history_gist_id)
    skip_list, reject_lines = get_blocked_titles(history)
    print(f"📋 {len(skip_list)} skipped, {len(reject_lines)} permanently rejected")

    suggestions = generate_suggestions(context, skip_list, reject_lines)
    for i, s in enumerate(suggestions):
        print(f"  {i+1}. [{s['category']}] {s['title']}")

    gist_id = store_suggestions_gist(suggestions, gh_token)
    print(f"💾 Stored in gist: {gist_id}")

    skip_url = f"{worker_url}/skip?token={gist_id}"
    subject, html_body = build_email(suggestions, worker_url, gist_id, skip_url)

    send_email(
        to=os.environ["NOTIFY_EMAIL"],
        subject=subject,
        html_body=html_body,
        gmail_user=os.environ["GMAIL_USER"],
        gmail_password=os.environ["GMAIL_APP_PASSWORD"],
    )
    print(f"📧 Email sent → {os.environ['NOTIFY_EMAIL']}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
