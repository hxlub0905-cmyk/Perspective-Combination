"""
implement.py — AI implementation engine for Fusi³

Flow:
1. Parse suggestion from SUGGESTION_JSON env var
2. Read source files that need modification
3. Call Claude to implement the changes
4. Create a new branch, commit, push
5. Open a GitHub PR
6. Send completion email with PR link
"""

import os
import json
import datetime
import subprocess
import requests
import anthropic
from pathlib import Path
from email_utils import send_email


# ─── File reader ──────────────────────────────────────────────────────────────

MAX_FILE_CHARS = 60_000  # ~15k tokens; dialog.py is 7600 lines ≈ 280k chars → truncate


def read_files_for_impl(files_affected: list[str]) -> dict[str, str]:
    """Read full contents of files that need modification, plus design context.

    Large files (> MAX_FILE_CHARS) are truncated: first 400 lines + all
    def/class signatures so Claude knows the full structure without blowing
    the context window.
    """
    root = Path(".")
    out: dict[str, str] = {}

    for filepath in files_affected:
        p = root / filepath
        if p.exists():
            raw = p.read_text(encoding="utf-8")
            if len(raw) > MAX_FILE_CHARS:
                lines = raw.splitlines()
                header = "\n".join(lines[:400])
                sigs = [l.rstrip() for l in lines if l.lstrip().startswith(("def ", "class "))]
                raw = (
                    f"# ⚠️ File truncated ({len(lines)} lines). First 400 lines + all signatures shown.\n\n"
                    + header
                    + "\n\n# ── All def/class signatures ──\n"
                    + "\n".join(sigs)
                )
                print(f"  ✂️  Truncated large file: {filepath} ({len(lines)} lines)")
            out[filepath] = raw
        else:
            print(f"  ⚠️  File not found: {filepath}")

    # Always include design tokens as context (affects UI changes)
    design = root / "perscomb/ui/design_tokens.py"
    if design.exists() and "perscomb/ui/design_tokens.py" not in out:
        out["[context] perscomb/ui/design_tokens.py"] = design.read_text(encoding="utf-8")

    # Include PROJECT_SUMMARY for broader context
    summary = root / "PROJECT_SUMMARY.md"
    if summary.exists():
        out["[context] PROJECT_SUMMARY.md"] = summary.read_text(encoding="utf-8")

    return out


# ─── Claude implementation ────────────────────────────────────────────────────

def implement_with_claude(suggestion: dict, file_contents: dict[str, str]) -> dict:
    """
    Ask Claude to implement the suggestion.

    Returns a dict with:
      commit_message, pr_title, pr_body,
      changes: [{file, action, content}]
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    files_block = ""
    for name, content in file_contents.items():
        lang = "python" if name.endswith(".py") else "markdown"
        files_block += f"\n\n### {name}\n```{lang}\n{content}\n```"

    core_note = ""
    if suggestion.get("requires_core_review"):
        core_note = (
            "\n⚠️ 此建議涉及核心演算法，必須：\n"
            "- 保持所有現有 public API 介面不變\n"
            "- 只做最小必要修改\n"
            "- 在 PR body 詳細說明每個改動的理由\n"
        )

    prompt = f"""你是 Fusi³ 專案的資深 Python 工程師。請根據以下建議實作程式碼修改。

## 建議
標題：{suggestion['title']}
類別：{suggestion['category']}
說明：{suggestion['description']}
實作提示：{suggestion.get('implementation_hint', '')}
{core_note}

## 現有程式碼
{files_block}

## 實作要求
1. 完整實作建議描述的功能，不留 TODO
2. 保持現有程式碼風格（縮排、命名慣例、docstring 格式）
3. UI 修改使用 design_tokens.py 的 Colors / Typography 等常數
4. 不破壞任何現有功能

## 輸出格式
只輸出以下 JSON，不要其他文字：
{{
  "commit_message": "type: short description in English (e.g. feat: add onboarding tooltip)",
  "pr_title": "PR 標題（中英混合）",
  "pr_body": "## 改動說明\\n\\n（繁體中文 Markdown，說明做了什麼、為什麼、如何測試）",
  "changes": [
    {{
      "file": "perscomb/ui/dialog.py",
      "action": "modify",
      "content": "完整的新版檔案內容（完整內容，非 diff）"
    }}
  ]
}}"""

    print("  🧠 Calling Claude API...")
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    # Try parsing; if truncated JSON, ask Claude to fix it
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error: {e} — retrying with fix prompt...")
        fix_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    "The JSON you returned was truncated or malformed. "
                    "Please output ONLY the complete, valid JSON object. "
                    "No markdown fences, no explanation."
                )},
            ],
        )
        fixed = fix_msg.content[0].text.strip()
        if fixed.startswith("```"):
            fixed = fixed.split("```")[1]
            if fixed.startswith("json"):
                fixed = fixed[4:]
        return json.loads(fixed.strip())


# ─── Apply changes ────────────────────────────────────────────────────────────

def apply_changes(changes: list[dict]) -> list[str]:
    """Write modified file contents to disk. Returns list of modified paths."""
    modified = []
    root = Path(".")
    for change in changes:
        filepath = change["file"]
        # Skip context-only entries
        if filepath.startswith("[context]"):
            continue
        p = root / filepath
        p.parent.mkdir(parents=True, exist_ok=True)
        action = change.get("action", "modify")
        if action in ("modify", "create"):
            p.write_text(change["content"], encoding="utf-8")
            print(f"  ✏️  {action}: {filepath}")
            modified.append(filepath)
        elif action == "delete" and p.exists():
            p.unlink()
            print(f"  🗑️  deleted: {filepath}")
            modified.append(filepath)
    return modified


# ─── Git helpers ──────────────────────────────────────────────────────────────

def run(cmd: str, check: bool = True) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nSTDERR: {result.stderr}")
    return result.stdout.strip()


def slugify(text: str, max_len: int = 35) -> str:
    slug = ""
    for ch in text.lower():
        if ch.isalnum():
            slug += ch
        elif ch in (" ", "_", "-") and not slug.endswith("-"):
            slug += "-"
    return slug.strip("-")[:max_len]


# ─── GitHub API ───────────────────────────────────────────────────────────────

def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Fusi3-AI-Bot",
    }


def open_pr(branch: str, title: str, body: str, gh_token: str, repo: str) -> str:
    """Open a GitHub PR and return its URL."""
    headers = _gh_headers(gh_token)
    r = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        headers=headers,
        json={"title": title, "body": body, "head": branch, "base": "main"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["html_url"]


# ─── Email ────────────────────────────────────────────────────────────────────

def send_completion_email(suggestion: dict, pr_url: str, branch: str):
    today = datetime.date.today().strftime("%Y年%m月%d日")
    core_badge = ""
    if suggestion.get("requires_core_review"):
        core_badge = """
      <div style="background:#fff8e1;border:1px solid #f0c040;border-radius:8px;padding:12px 16px;margin:16px 0;">
        <strong>⚠️ 核心演算法異動</strong> — 請仔細審核再 merge
      </div>"""

    subject = f"[Fusi³ AI] ✅ 實作完成 — {suggestion['title']}"
    html_body = f"""<html><body style="font-family:-apple-system,sans-serif;max-width:600px;margin:0 auto;padding:20px;color:#1a1a1a;">
<div style="background:#22c55e;color:white;padding:16px 20px;border-radius:10px 10px 0 0;">
  <h2 style="margin:0;font-size:18px;">✅ AI 實作完成！</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px;">{today}</p>
</div>
<div style="border:1px solid #e5e5e5;border-top:none;border-radius:0 0 10px 10px;padding:24px;">
  <h3 style="margin:0 0 8px;">{suggestion['title']}</h3>
  <p style="color:#555;margin:0 0 16px;">{suggestion['description']}</p>
  {core_badge}
  <a href="{pr_url}" style="background:#FF6B35;color:white;padding:14px 28px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px;display:inline-block;">🔍 Review PR on GitHub</a>
  <p style="color:#aaa;font-size:12px;margin-top:12px;">滿意後在 GitHub merge 即可。</p>
  <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
  <p style="color:#bbb;font-size:11px;">Branch: <code>{branch}</code></p>
</div>
</body></html>"""

    send_email(
        to=os.environ["NOTIFY_EMAIL"],
        subject=subject,
        html_body=html_body,
        gmail_user=os.environ["GMAIL_USER"],
        gmail_password=os.environ["GMAIL_APP_PASSWORD"],
    )
    print(f"📧 Completion email sent → {os.environ['NOTIFY_EMAIL']}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🤖 Fusi³ AI implement.py starting...")

    gh_token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["REPO_FULL_NAME"]
    suggestion = json.loads(os.environ["SUGGESTION_JSON"])
    print(f"📋 Suggestion: [{suggestion.get('category')}] {suggestion.get('title')}")

    # Git identity
    run('git config user.name "Fusi³ AI Bot"')
    run('git config user.email "ai-bot@fusi3-bot.dev"')

    # Create branch
    date_str = datetime.date.today().strftime("%Y%m%d")
    slug = slugify(suggestion.get("title", "update"))
    branch = f"ai/{date_str}-{slug}"
    run(f"git checkout -b {branch}")
    print(f"🌿 Branch: {branch}")

    # Read relevant files
    files_affected = suggestion.get("files_affected", [])
    file_contents = read_files_for_impl(files_affected)
    print(f"📁 Loaded {len(file_contents)} files")

    # Call Claude to implement
    result = implement_with_claude(suggestion, file_contents)

    # Apply changes to disk
    print("✏️  Applying changes...")
    modified = apply_changes(result["changes"])
    if not modified:
        raise RuntimeError("No files were modified — aborting.")

    # Syntax check — catch obvious errors before committing
    print("🔍 Syntax checking modified Python files...")
    syntax_errors = []
    for f in modified:
        if f.endswith(".py"):
            check = subprocess.run(
                f'python -m py_compile "{f}"',
                shell=True, capture_output=True, text=True
            )
            if check.returncode != 0:
                syntax_errors.append(f"{f}: {check.stderr.strip()}")
                print(f"  ❌ Syntax error: {f}\n     {check.stderr.strip()}")
            else:
                print(f"  ✅ OK: {f}")

    if syntax_errors:
        raise RuntimeError(
            "Syntax errors found — aborting commit:\n" + "\n".join(syntax_errors)
        )
    print("✅ All files passed syntax check")

    # Commit & push
    for f in modified:
        run(f'git add "{f}"')
    run(f'git commit -m "{result["commit_message"]}"')
    run(f"git push https://x-access-token:{gh_token}@github.com/{repo}.git {branch}")
    print(f"⬆️  Pushed: {branch}")

    # Build PR body
    pr_body = result["pr_body"]
    if suggestion.get("requires_core_review"):
        pr_body += (
            f"\n\n---\n> ⚠️ **核心演算法異動** — 請手動仔細審核再 merge\n"
            f"> 原因：{suggestion.get('core_reason', '')}"
        )
    pr_body += "\n\n---\n> 🤖 Auto-generated by [Fusi³ AI Bot](https://github.com/features/actions)"

    # Open PR
    pr_url = open_pr(branch, result["pr_title"], pr_body, gh_token, repo)
    print(f"🔗 PR: {pr_url}")

    # Notify
    send_completion_email(suggestion, pr_url, branch)
    print("✅ All done!")


if __name__ == "__main__":
    main()
