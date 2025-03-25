import os
from email import policy
from email.parser import BytesParser

def load_eml(file_path: str):
    """Load an email file and parse it."""
    with open(file_path, 'rb') as f:
        return BytesParser(policy=policy.default).parse(f)

def extract_text_from_eml(msg):
    """Extract plain text from an email message."""
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == 'text/plain':
                return part.get_payload(decode=True).decode()
    else:
        return msg.get_payload(decode=True).decode()
    return ""