"""
notify_error.py — Send error notification email when implement.py fails.
Called by GitHub Actions `if: failure()` step.
"""

import os
import json
import datetime
from email_utils import send_email


def main():
    suggestion = json.loads(os.environ.get("SUGGESTION_JSON", "{}"))
    run_url = os.environ.get("RUN_URL", "https://github.com")
    today = datetime.date.today().strftime("%Y年%m月%d日")

    subject = f"[Fusi³ AI] ❌ 實作失敗 — {suggestion.get('title', 'Unknown')}"
    html_body = f"""<html><body style="font-family:-apple-system,sans-serif;max-width:600px;margin:0 auto;padding:20px;color:#1a1a1a;">
<div style="background:#ef4444;color:white;padding:16px 20px;border-radius:10px 10px 0 0;">
  <h2 style="margin:0;font-size:18px;">❌ AI 實作失敗</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px;">{today}</p>
</div>
<div style="border:1px solid #e5e5e5;border-top:none;border-radius:0 0 10px 10px;padding:24px;">
  <h3 style="margin:0 0 8px;">{suggestion.get('title', 'Unknown suggestion')}</h3>
  <p style="color:#555;margin:0 0 20px;">AI 在實作此建議時發生錯誤，無法完成。請查看 GitHub Actions log 了解詳細原因。</p>
  <a href="{run_url}" style="background:#6b7280;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px;display:inline-block;">🔍 查看 Actions Log</a>
  <p style="color:#aaa;font-size:12px;margin-top:16px;">你可以：手動修復後重新觸發，或等明天 AI 提新建議。</p>
</div>
</body></html>"""

    send_email(
        to=os.environ["NOTIFY_EMAIL"],
        subject=subject,
        html_body=html_body,
        gmail_user=os.environ["GMAIL_USER"],
        gmail_password=os.environ["GMAIL_APP_PASSWORD"],
    )
    print("📧 Error notification sent.")

    # LINE notification
    from line_utils import send_line_message, format_error_message
    send_line_message(format_error_message(suggestion, run_url))


if __name__ == "__main__":
    main()
