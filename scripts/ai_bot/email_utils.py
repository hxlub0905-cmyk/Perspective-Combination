"""email_utils.py — Simple Gmail SMTP helper."""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(
    to: str,
    subject: str,
    html_body: str,
    gmail_user: str,
    gmail_password: str,
) -> None:
    """Send an HTML email via Gmail SMTP (SSL port 465)."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Fusi\u00b3 AI Bot <{gmail_user}>"
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, [to], msg.as_string())
