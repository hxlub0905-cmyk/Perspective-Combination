"""
suggest.py — Daily AI suggestion generator for Fusi³

Flow:
1. Read project source files
2. Load skip history from GitHub Gist
3. Call Claude to generate 1 focused suggestion
4. Store suggestion in a new GitHub Gist (gist ID = token)
5. Send email with OK / Skip links pointing to Cloudflare Worker
"""

import os
import json
import uuid
import datetime
import requests
import anthropic
from pathlib import Path
from email_utils import send_email


# ─── Project reader ──────────────────────────────────────────────────────────

def read_project_context() -> dict:
    """Read key files from the checked-out repo."""
    root = Path(".")
    context: dict[str, str] = {}

    # Full files (small enough to send entirely)
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

    # perspective_combine.py — first 600 lines + all signatures
    pc_path = root / "perscomb/core/perspective_combine.py"
    if pc_path.exists():
        lines = pc_path.read_text(encoding="utf-8").splitlines()
        preview = "\n".join(lines[:600])
        sigs = [l.rstrip() for l in lines if l.lstrip().startswith(("def ", "class "))]
        context["perscomb/core/perspective_combine.py"] = (
            preview + "\n\n# --- remaining signatures ---\n" + "\n".join(sigs[30:])
        )

    # dialog.py — structure only (file is 7600+ lines)
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


# ─── Skip history ─────────────────────────────────────────────────────────────

def load_history(gh_token: str, gist_id: str) -> list:
    """Fetch suggestion history from a private GitHub Gist."""
    if not gist_id:
        return []
    headers = _gh_headers(gh_token)
    try:
        r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=10)
        if r.ok:
            raw = r.json().get("files", {}).get("history.json", {}).get("content", "[]")
            return json.loads(raw)
    except Exception as e:
        print(f"⚠️  Could not load history: {e}")
    return []


def skipped_titles(history: list) -> list[str]:
    """Return titles that were skipped less than 30 days ago."""
    today = datetime.date.today()
    result = []
    for item in history:
        if item.get("action") != "skip":
            continue
        again_after = item.get("suggest_again_after", "")
        if again_after:
            try:
                cutoff = datetime.date.fromisoformat(again_after[:10])
                if today < cutoff:
                    result.append(item["title"])
            except ValueError:
                pass
    return result


# ─── Claude call ─────────────────────────────────────────────────────────────

def generate_suggestion(context: dict, skip_titles: list[str]) -> dict:
    """Call Claude API and get one structured suggestion."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    files_block = ""
    for name, content in context.items():
        files_block += f"\n\n### {name}\n```\n{content}\n```"

    skip_block = ""
    if skip_titles:
        skip_block = "\n\n已略過（本月內請勿重複提）：\n" + "\n".join(f"- {t}" for t in skip_titles)

    prompt = f"""你是 Fusi³ 專案的 AI 開發助理。Fusi³ 是一個給半導體工程師用的 SEM 影像融合分析桌面工具（PySide6）。

請分析以下程式碼，提出**一個**今日最值得做的改善建議。
{files_block}
{skip_block}

優先順序：
1. UX 改善 / 新手引導（這個工具對新用戶不直觀）
2. Bug 修復 / 穩定性
3. 新功能開發

規則：
- 只提一個建議，必須具體且可在 1-3 小時內實作完成
- 若建議會修改核心演算法（ebeam_snr.py、perspective_combine.py、roi_set.py），requires_core_review = true
- files_affected 只列真正需要修改的檔案

只輸出以下 JSON，不要其他文字：
{{
  "title": "簡短標題（中英混合，15字內）",
  "category": "UX | Bug | Feature | Refactor",
  "description": "詳細說明（繁體中文，100字內）",
  "motivation": "Why this matters for users (English, 80 words max)",
  "files_affected": ["perscomb/ui/dialog.py"],
  "estimated_complexity": "small | medium | large",
  "requires_core_review": false,
  "core_reason": "",
  "implementation_hint": "具體說明要改哪個 function / class，怎麼改（80字內，給 AI 工程師看的）"
}}"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ─── Gist storage ─────────────────────────────────────────────────────────────

def store_suggestion_gist(suggestion: dict, gh_token: str) -> str:
    """Create a private Gist with the suggestion. Returns the gist ID (used as token)."""
    headers = _gh_headers(gh_token)
    payload = {
        "description": f"Fusi³ AI Suggestion – {suggestion['title']}",
        "public": False,
        "files": {
            "suggestion.json": {
                "content": json.dumps(suggestion, ensure_ascii=False, indent=2)
            }
        },
    }
    r = requests.post("https://api.github.com/gists", headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["id"]


# ─── Email ────────────────────────────────────────────────────────────────────

def build_email(suggestion: dict, approve_url: str, skip_url: str) -> tuple[str, str]:
    """Return (subject, html_body)."""
    cat_emoji = {"UX": "🎨", "Bug": "🐛", "Feature": "✨", "Refactor": "🔧"}.get(
        suggestion["category"], "💡"
    )
    complexity_color = {"small": "#22c55e", "medium": "#f59e0b", "large": "#ef4444"}.get(
        suggestion["estimated_complexity"], "#6b7280"
    )
    today = datetime.date.today().strftime("%Y年%m月%d日")

    core_block = ""
    if suggestion.get("requires_core_review"):
        core_block = f"""
      <div style="background:#fff8e1;border:1px solid #f0c040;border-radius:8px;padding:12px 16px;margin:16px 0;">
        <strong>⚠️ 核心演算法異動</strong><br>
        <span style="color:#555;">{suggestion.get('core_reason','')}</span><br>
        <small style="color:#888;">需手動審核才 merge</small>
      </div>"""

    files_html = "".join(
        f'<code style="display:block;background:#f4f4f4;padding:3px 8px;border-radius:4px;font-size:12px;margin:2px 0;">{f}</code>'
        for f in suggestion["files_affected"]
    )

    subject = f"[Fusi³ AI] {cat_emoji} {suggestion['title']}"
    body = f"""<html><body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:600px;margin:0 auto;padding:20px;color:#1a1a1a;">

<div style="background:#FF6B35;color:white;padding:16px 20px;border-radius:10px 10px 0 0;">
  <h2 style="margin:0;font-size:18px;">🤖 Fusi³ AI 今日開發建議</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px;">{today}</p>
</div>

<div style="border:1px solid #e5e5e5;border-top:none;border-radius:0 0 10px 10px;padding:24px;">

  <div style="background:#f9fafb;border-left:4px solid #FF6B35;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:20px;">
    <span style="background:#FF6B35;color:white;font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600;">{suggestion['category']}</span>
    <span style="background:#f0f0f0;color:{complexity_color};font-size:11px;padding:2px 8px;border-radius:20px;margin-left:6px;font-weight:600;">{suggestion['estimated_complexity']}</span>
    <h3 style="margin:10px 0 0;font-size:20px;">{cat_emoji} {suggestion['title']}</h3>
  </div>

  <h4 style="color:#444;margin:0 0 6px;">📋 建議說明</h4>
  <p style="color:#333;line-height:1.7;margin:0 0 16px;">{suggestion['description']}</p>

  <h4 style="color:#444;margin:0 0 6px;">🎯 Why it matters</h4>
  <p style="color:#666;line-height:1.7;font-style:italic;margin:0 0 16px;">{suggestion['motivation']}</p>

  <h4 style="color:#444;margin:0 0 6px;">📁 Files to modify</h4>
  <div style="margin:0 0 16px;">{files_html}</div>

  {core_block}

  <div style="margin-top:28px;display:flex;gap:12px;">
    <a href="{approve_url}" style="background:#22c55e;color:white;padding:14px 36px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px;display:inline-block;">✅ OK，幫我做！</a>
    <a href="{skip_url}" style="background:#f5f5f5;color:#666;padding:14px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:16px;display:inline-block;">⏭️ Skip</a>
  </div>
  <p style="color:#aaa;font-size:12px;margin-top:10px;">
    點 OK → AI 自動實作 → 開 PR → 寄通知給你 → 你晚上 review merge<br>
    點 Skip → 記錄略過，一個月後若未完成會再提
  </p>

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

    # 1. Read code
    context = read_project_context()
    print(f"📁 Read {len(context)} context sections")

    # 2. Load history
    history = load_history(gh_token, history_gist_id)
    skip_list = skipped_titles(history)
    print(f"📋 {len(skip_list)} titles on skip cooldown")

    # 3. Generate suggestion
    suggestion = generate_suggestion(context, skip_list)
    print(f"💡 Suggestion: [{suggestion['category']}] {suggestion['title']}")

    # 4. Store in gist → gist ID becomes the approval token
    gist_id = store_suggestion_gist(suggestion, gh_token)
    print(f"💾 Stored in gist: {gist_id}")

    # 5. Build URLs
    approve_url = f"{worker_url}/approve?token={gist_id}"
    skip_url = f"{worker_url}/skip?token={gist_id}"

    # 6. Send email
    subject, html_body = build_email(suggestion, approve_url, skip_url)
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
