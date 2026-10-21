"""Readiness, instance ownership and cleanup helpers for the Mac launcher."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import build_opener, ProxyHandler, Request
from mac_stack_config import ROOT, PYTHON, PORTS, HEALTH, required_files

RUN = ROOT / ".run/mac-stack"
SCRIPT = str(ROOT / "scripts/mac_stack.py")


def request(url, post=False, timeout=2):
    request = Request(url, data=b"" if post else None)
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.15):
            return True
    except OSError:
        return False


def fingerprint(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="], capture_output=True, text=True, env=dict(os.environ, LC_ALL="C"))
    return result.stdout.strip()


def owned(record):
    try:
        current = fingerprint(int(record["pid"]))
        return SCRIPT in current and current == record["fingerprint"]
    except (KeyError, ValueError):
        return False


def records():
    RUN.mkdir(parents=True, exist_ok=True)
    output = {}
    for path in RUN.glob("*.json"):
        try:
            output[path.stem] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return output


@contextmanager
def instance(name):
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / (name + ".json")
    with (RUN / "registry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        live = {key: record for key, record in records().items() if owned(record)}
        if name in live or "all" in live or (name == "all" and live):
            raise RuntimeError("已有本启动器运行：" + ", ".join(live) + "；先执行 scripts/stop_all.sh")
        record = {"pid": os.getpid(), "fingerprint": fingerprint(os.getpid()), "name": name}
        path.write_text(json.dumps(record))
        fcntl.flock(lock, fcntl.LOCK_UN)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()).get("pid") == os.getpid():
            path.unlink()


def stop_instances():
    found = False
    for name, record in records().items():
        if owned(record):
            print("停止", name, flush=True)
            try:
                os.kill(record["pid"], signal.SIGTERM)
                found = True
            except ProcessLookupError:
                pass
    if not found:
        print("没有本启动器管理的服务；旧版脚本需在原终端 Ctrl-C。")
        return
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if not any(owned(record) for record in records().values()):
            print("受管服务已停止。")
            return
        time.sleep(.25)
    raise RuntimeError("停止等待超时；检查对应日志，不会强杀无关进程")


def preflight(names, asr, tse, imports=True):
    import shutil
    errors = []
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        errors.append(f"缺少 Python 环境：{PYTHON}")
    for name, paths in required_files(names, asr, tse).items():
        missing = [str(path) for path in paths if not path.is_file() or not path.stat().st_size]
        print(("缺失" if missing else "文件就绪") + ": " + name, flush=True)
        errors.extend(missing)
    for component, command in (("ros", "docker"), ("memory", "cargo")):
        if component in names and not shutil.which(command):
            errors.append("缺少命令：" + command)
    if errors:
        raise RuntimeError("启动检查失败：\n" + "\n".join(errors))
    if imports and "brain" in names:
        voices = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True, check=True).stdout
        if not any(line.startswith("Tingting ") for line in voices.splitlines()):
            raise RuntimeError("缺少 macOS Tingting 中文声音，请先在系统声音设置安装")
    if imports and any(name in names for name in ("vision", "voice", "attention")):
        modules = set()
        if "voice" in names:
            modules.update(("sounddevice", "sherpa_onnx", "scipy", "librosa", "torch", "torchaudio", "transformers", "websocket"))
        if "voice" in names and asr in {"whisper", "vosk"}:
            modules.add("vosk")
        if "vision" in names:
            modules.update(("cv2", "mediapipe", "aiohttp"))
        if "attention" in names:
            modules.update(("robot_attention_perception", "numpy", "websocket"))
        code = "import importlib; [importlib.import_module(n) for n in " + repr(sorted(modules)) + "]; print('Python dependencies OK')"
        subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, check=True)


def wait_ready(name, process=None, timeout=180):
    deadline = time.monotonic() + timeout
    last = "未监听"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{name} 已退出（{process.returncode}）；查看 {RUN / (name + '.log')}")
        try:
            if name == "ros":
                if port_open(PORTS[name]):
                    return
            else:
                data = request(f"http://127.0.0.1:{PORTS[name]}{HEALTH[name]}")
                if name == "qwen" and data.get("status") != "ok":
                    raise RuntimeError("模型仍在加载")
                if name == "vision" and not data.get("state"):
                    raise RuntimeError("等待摄像头画面/授权")
                if name == "attention":
                    health = data.get("diagnostics") or {}
                    if not all(health.get(key) for key in ("vision_ok", "voice_ok", "rosbridge_connected")):
                        raise RuntimeError("等待视觉、声音和 ROS bridge 连接")
                return
        except Exception as exc:
            last = str(exc)
        time.sleep(.25)
    raise RuntimeError(f"{name} 就绪超时：{last}；查看 {RUN / (name + '.log')}")


def stop_processes(processes):
    # Every child has its own process group, including model/ASR grandchildren.
    for name, process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
