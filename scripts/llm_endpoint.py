"""Explicit local/remote model ownership; remote endpoints are never stopped locally."""
import os,json
from pathlib import Path
from urllib.parse import urlparse
DEFAULT='http://172.16.60.162:18050/v1/chat/completions'
def endpoint():return os.environ.get('BRAIN_LLM_BASE_URL') or DEFAULT
def remote():return urlparse(endpoint()).hostname not in {'localhost','127.0.0.1','::1'}
def health_url():
    p=urlparse(endpoint());return f'{p.scheme}://{p.netloc}/health'
