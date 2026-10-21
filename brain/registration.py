"""Deterministic opt-in registration intent; never inferred by the language model."""
import re


def parse_registration(text):
    compact=re.sub(r'[\s，,。.!！?？：:、]', '', str(text))
    # A direct request is required; quoting somebody's introduction is not enrollment.
    direct=re.match(r'^(?:你好|喂|嘿)?(?:小圆|小园|小元|小袁|机器人)',compact)
    if not direct:
        return None
    body=compact[direct.end():]
    if not re.search(r'(?:请记住我|记住我|帮我注册|我要注册|请注册|注册一下)',body):
        return None
    name=re.search(r'(?:我叫|我的名字是)([\u4e00-\u9fffA-Za-z0-9_-]{1,32}?)(?=请|帮我|我要|记住我|我是|注册|$)',body)
    if not name:
        return {'name':'','role':'known'}
    return {'name':name.group(1),'role':'owner' if re.search(r'我是(?:你的)?主人',body) else 'known'}
