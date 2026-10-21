"""哪个模型在当慢脑，以及切换它。

Two on offer. The local Qwen on the LAN machine keeps every sentence inside
the room, which is why it is the default. DeepSeek's API is a stronger model
and the conversation leaves this machine to reach it — that is a choice the
person in front of the robot is making, so the console says it in as many
words rather than switching quietly.

The key is never in this file, never in the state the panel reads, and never
in a log line. It comes from DEEPSEEK_API_KEY or from a file named by
BRAIN_DEEPSEEK_KEY_FILE, and everything downstream only ever learns whether
there is one.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# llama.cpp takes chat_template_kwargs and needs it: without
# enable_thinking=false, Qwen3.5 spends its whole token budget on a
# reasoning_content block and returns no answer at all (measured: 200
# completion tokens, empty content). A cloud API rejects unknown fields, so
# the flag travels with the backend rather than being sent to everybody.
BACKENDS: dict[str, dict[str, Any]] = {
    "qwen_162": {
        "label": "本地 Qwen3.5-9B（172.16.60.162）",
        "where": "局域网内的常驻 llama-server，对话内容不出本地网络",
        "base_url_env": "BRAIN_LLM_BASE_URL",
        "model_env": "BRAIN_LLM_MODEL",
        "model_default": "qwen3.5-9b",
        "key_envs": (),
        "template_kwargs": True,
        "timeout_default": "60",
    },
    "deepseek": {
        "label": "DeepSeek V3（deepseek-chat，云端 API）",
        "where": "云端服务：对话内容会离开这台机器，发到 DeepSeek 的服务器",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url_default": "https://api.deepseek.com/v1",
        "model_env": "DEEPSEEK_MODEL",
        "model_default": "deepseek-chat",
        "key_envs": ("DEEPSEEK_API_KEY",),
        "key_file_env": "BRAIN_DEEPSEEK_KEY_FILE",
        "template_kwargs": False,
        "timeout_default": "30",
    },
}
DEFAULT_BACKEND = os.environ.get("BRAIN_LLM_BACKEND", "qwen_162")


def _api_key(spec: dict[str, Any]) -> str:
    for name in spec.get("key_envs") or ():
        value = os.environ.get(name, "").strip()
        if value:
            return value
    path = os.environ.get(spec.get("key_file_env") or "", "").strip()
    if path:
        try:
            return Path(path).expanduser().read_text(encoding="utf-8").strip().splitlines()[0].strip()
        except (OSError, IndexError):
            return ""
    return ""


def resolve(name: str) -> dict[str, Any]:
    """Everything needed to build a client, including the key. Never logged."""
    spec = BACKENDS.get(str(name))
    if not spec:
        raise KeyError(f"unknown slow-brain backend {name!r}")
    base_url = os.environ.get(spec.get("base_url_env") or "", "").strip() or str(spec.get("base_url_default") or "")
    return {"name": str(name), "label": spec["label"], "where": spec["where"],
            "base_url": base_url,
            "model": os.environ.get(spec.get("model_env") or "", "").strip() or str(spec["model_default"]),
            "api_key": _api_key(spec),
            "template_kwargs": bool(spec["template_kwargs"]),
            "timeout": float(os.environ.get("BRAIN_LLM_TIMEOUT", spec["timeout_default"]))}


def describe() -> list[dict[str, Any]]:
    """What the console shows: never the key, only whether there is one."""
    rows = []
    for name, spec in BACKENDS.items():
        try:
            resolved = resolve(name)
        except KeyError:
            continue
        needs_key = bool(spec.get("key_envs") or spec.get("key_file_env"))
        has_key = bool(resolved["api_key"])
        if not resolved["base_url"]:
            reason = f"没有配置地址（{spec.get('base_url_env')}）"
        elif needs_key and not has_key:
            reason = f"没有找到 API key（{'/'.join(spec.get('key_envs') or ())} 或 {spec.get('key_file_env')}）"
        else:
            reason = ""
        rows.append({"id": name, "label": resolved["label"], "where": resolved["where"],
                     "model": resolved["model"], "endpoint": resolved["base_url"],
                     "needs_key": needs_key,
                     "key_present": has_key if needs_key else None,
                     "available": not reason, "reason": reason})
    return rows
