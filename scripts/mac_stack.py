#!/usr/bin/env python3
"""Start the complete Mac stack or one component using the same configuration."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from urllib.parse import urlencode
from mac_stack_config import ROOT, ORDER, PORTS, PYTHON, COMPOSE, specs, environment
from llm_endpoint import remote, endpoint, health_url
from mac_stack_support import RUN, instance, records, owned, stop_instances, preflight, port_open, request, wait_ready, stop_processes


def start_capture(args):
    query = urlencode({"profile": args.profile, "device": args.device, "tse": args.tse, "seconds": 0})
    data = request("http://127.0.0.1:8090/api/start?" + query, post=True, timeout=120)
    if not data.get("running") and data.get("capture_request"):
        # The array is not on the bus right now. The request is a standing one:
        # the voice service retries every few seconds and comes back on its own
        # when the device is plugged in again, so taking the whole stack down
        # for it means nothing else works either while somebody finds the cable.
        print("麦克风暂时不可用，采集请求已保留，插上后会自动恢复：\n  " + str(data.get("error") or ""),
              flush=True)
        return
    if not data.get("running") or data.get("error"):
        raise RuntimeError("麦克风启动失败：" + str(data.get("error") or data))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        data = request("http://127.0.0.1:8090/api/state")
        scene = data.get("audio_scene") or {}
        error = scene.get("error") or data.get("asr_error") or data.get("speaker_model_error")
        if error:
            raise RuntimeError("声音模型初始化失败：" + str(error))
        if data.get("asr_model_ready") and data.get("speaker_model_ready") and scene.get("classifier_ready"):
            print("麦克风、声纹、音乐/曲风模型已就绪", flush=True)
            return
        time.sleep(.3)
    raise RuntimeError("声音模型加载超时；查看 voice.log 和8090面板")


def docker_ready(env):
    check = lambda: subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if check():
        return
    print("启动 Docker Desktop…", flush=True)
    subprocess.run(["open", "-a", "Docker"], check=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(1)
    raise RuntimeError("Docker Desktop 尚未就绪")


def run(args):
    names = list(ORDER) if args.component == "all" else [args.component]
    if args.component=="all" and remote():names.remove("qwen")
    configured = specs(args.session, args.asr, args.camera)
    preflight(names, args.asr, args.tse, imports=not args.dry_run)
    if args.dry_run:
        if remote():print("remote Qwen: "+endpoint()+" (not managed locally)")
        for name in names:
            print(name + ": " + shlex.join(COMPOSE + ["up", "-d", "--build"] if name == "ros" else configured[name].command))
        return
    if args.component in {"all","brain"} and remote():
        result=request(health_url(),timeout=10)
        if result.get("status")!="ok":raise RuntimeError("远端Qwen未就绪："+str(result))
        print("使用远端Qwen："+endpoint(),flush=True)
    with instance(args.component):
        for name in names:
            if port_open(PORTS[name]):
                raise RuntimeError(f"{name} 端口 {PORTS[name]} 已被占用；停止旧实例后重试")
        env = environment(args.session, args.asr)
        if "ros" in names:
            docker_ready(env)
        dependencies = {"brain": ("qwen", "memory", "attention"), "attention": ("ros", "memory", "vision", "voice")}
        if args.component != "all":
            for dependency in dependencies.get(args.component, ()):
                if dependency=="qwen" and remote():continue
                wait_ready(dependency, timeout=30)
        processes, logs = [], []
        owns_ros = False
        try:
            for name in names:
                print("启动 " + name + f"（{PORTS[name]}）…", flush=True)
                log = (RUN / (name + ".log")).open("w")
                logs.append(log)
                if name == "ros":
                    owns_ros = True
                    process = subprocess.Popen(COMPOSE + ["up", "-d", "--build"], cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append(("ros-build", process))
                    if process.wait() != 0:
                        raise RuntimeError("ROS容器启动失败；查看 ros.log")
                    processes.remove(("ros-build", process))
                    wait_ready(name)
                else:
                    spec = configured[name]
                    process = subprocess.Popen(spec.command, cwd=spec.cwd, env=spec.env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append((name, process))
                    wait_ready(name, process)
                if name == "voice" and not args.no_capture:
                    start_capture(args)
                print(name + " 已就绪", flush=True)
            if not args.no_browser:
                pages = ["attention"] if args.component == "all" else [name for name in names if name in {"vision", "voice", "attention", "brain"}]
                for name in pages:
                    subprocess.run(["open", f"http://127.0.0.1:{PORTS[name]}"], check=False)
            print(f"启动完成。日志：{RUN}\n按 Ctrl-C 停止本次启动的服务。", flush=True)
            while True:
                for name, process in processes:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} 已退出；查看 {name}.log")
                if owns_ros and not port_open(PORTS["ros"]):
                    raise RuntimeError("ROS bridge 已退出")
                time.sleep(.5)
        finally:
            stop_processes(processes)
            if owns_ros:
                subprocess.run(COMPOSE + ["down"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            for log in logs:
                log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("all", "check", "status", "stop") + ORDER)
    parser.add_argument("--session", default=os.environ.get("SESSION_ID", "mac-conversation-test"))
    parser.add_argument("--asr", choices=("sherpa", "whisper", "vosk"), default="sherpa")
    parser.add_argument("--tse", choices=("baseline", "speakerbeam", "multi"), default="baseline")
    parser.add_argument("--profile", default="mac_builtin")
    parser.add_argument("--device", default="MacBook Pro麦克风")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, terminate)
    try:
        if args.component == "stop":
            stop_instances()
        elif args.component == "status":
            print("受管实例：", ", ".join(name for name, record in records().items() if owned(record)) or "无")
            for name in ORDER:
                if name=="qwen" and remote():
                    try:print("qwen remote",endpoint(),request(health_url(),timeout=3).get("status","unknown"))
                    except Exception:print("qwen remote",endpoint(),"不可达")
                    continue
                print(name, PORTS[name], "监听中" if port_open(PORTS[name]) else "未启动")
        elif args.component == "check":
            preflight([name for name in ORDER if name!="qwen" or not remote()], args.asr, args.tse)
            print("模型和依赖检查通过；尚未开启摄像头/麦克风。")
        else:
            run(args)
    except KeyboardInterrupt:
        print("已停止本次启动。")
    except Exception as exc:
        print("启动失败：", str(exc), flush=True)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
