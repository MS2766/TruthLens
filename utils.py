# utils.py
import re

def clean_text(s):
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s).strip()
