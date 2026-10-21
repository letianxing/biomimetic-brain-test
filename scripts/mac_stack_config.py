"""One source of model paths, environment and commands for full/individual startup."""
from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
from llm_endpoint import endpoint

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
ATTENTION = WORKSPACE / "robot-attention-perception"
VOICE = WORKSPACE / "voice-detection"
VISION = WORKSPACE / "vision-detection"
MODELS = WORKSPACE / "robo-model"
PYTHON = ATTENTION / ".venv-mac/bin/python"
COMPOSE = ["docker", "compose", "-f", str(ATTENTION / "docker-compose.mac.yml")]
ORDER = ("qwen", "ros", "memory", "vision", "voice", "attention", "brain")
PORTS = dict(zip(ORDER, (18050, 9090, 8788, 8080, 8090, 8092, 8094)))
HEALTH = {"qwen": "/health", "memory": "/v1/health", "vision": "/api/state", "voice": "/api/state",
          "attention": "/api/state", "brain": "/api/health"}
SHERPA = VOICE / "weights/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
SENSEVOICE = VOICE / "weights/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
VOSK = VOICE / "weights/vosk-model-small-cn-0.22"
QWEN = MODELS / "models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf"
# 用户上传的 DeepSeek key 文件位置。文件内容从不进入代码、日志或面板状态。
DEEPSEEK_KEY_FILE = Path("/Applications/权重/deepseek API KEY.txt")


@dataclass
class Spec:
    command: list[str]
    cwd: Path
    env: dict[str, str]


def environment(session, asr="sherpa"):
    env = dict(os.environ)
    env.update(NO_PROXY="localhost,127.0.0.1,::1,172.16.60.162," + env.get("NO_PROXY", ""))
    env["no_proxy"] = env["NO_PROXY"]
    env.update(HOST="127.0.0.1", VISION_PORT="8080", VOICE_PORT="8090", CONSOLE_PORT="8092",
               MEMORY_PORT="8788", ROSBRIDGE_PORT="9090", ROS_DISTRO="jazzy", ROS_BASE_IMAGE="ros:jazzy-ros-base", SESSION_ID=session, PYTHONUNBUFFERED="1", ATTENTION_S2S_ENABLED="0",
               QWEN_35_9B_HOST="127.0.0.1", QWEN_35_9B_PORT="18050", QWEN_35_9B_ALIAS="qwen3.5-9b",
               QWEN_35_9B_MODEL_PATH=str(QWEN), QWEN_35_9B_CTX="4096", QWEN_35_9B_PARALLEL="1",
               QWEN_35_9B_GPU_LAYERS="99", QWEN_35_9B_LLAMA_SERVER_BIN=str(MODELS / "third_party/llama.cpp/build/bin/llama-server"),
               BRAIN_LLM_BASE_URL=endpoint(), BRAIN_LLM_MODEL="qwen3.5-9b",
               # 慢脑可以在面板上切到 DeepSeek。这里只传 key 文件的路径，不读内容、
               # 不写进代码或日志；Brain 需要时自己读，状态里只报「有没有」。
               BRAIN_DEEPSEEK_KEY_FILE=env.get("BRAIN_DEEPSEEK_KEY_FILE", str(DEEPSEEK_KEY_FILE)),
               BRAIN_LLM_API_KEY="", BRAIN_LLM_TIMEOUT=env.get("BRAIN_LLM_TIMEOUT", "60"),
               # The remote server now runs with n_ctx 32768 (was 4096, which the
               # field session overran twice). The prompt budget stays well under
               # it on purpose: it is a latency and relevance choice, not a limit.
               BRAIN_LLM_CONTEXT_TOKENS=env.get("BRAIN_LLM_CONTEXT_TOKENS", "8192"),
               # Where the robot is, for the weather lookup. Set by the user;
               # with no value the tool says it does not know rather than
               # reporting some other city's weather.
               BRAIN_LOCATION=env.get("BRAIN_LOCATION", "上海"),
               # How long a spoken reply may be. Six characters sounds like a
               # person and sometimes loses the substance — "同意你的看法" instead
               # of saying what to agree with. Eight to twelve is where both fit;
               # raise these two if the replies feel empty.
               BRAIN_REPLY_CHARS=env.get("BRAIN_REPLY_CHARS", "6"),
               BRAIN_REPLY_CHARS_MAX=env.get("BRAIN_REPLY_CHARS_MAX", "14"),
               BRAIN_VOICE_URL="http://127.0.0.1:8090", BRAIN_ROSBRIDGE_URL="ws://127.0.0.1:9090", VOICE_ROSBRIDGE_URL="ws://127.0.0.1:9090",
               VOICE_AUDIO_SCENE="1", VOICE_AUDIO_MODEL=str(VOICE / "weights/ast-audioset"),
               VOICE_GENRE_MODEL=str(VOICE / "weights/music-genre"),
               VOICE_SPEAKER_MODEL=str(VOICE / "weights/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"),
               VOICE_ASR_LANGUAGE="zh-CN", VOICE_ASR_SAMPLE_RATE="16000", VOICE_ASR_SHERPA_MODEL="",
               VOICE_ASR_VOSK_MODEL="", VOICE_ASR_COMMAND="", VOICE_ASR_SENSEVOICE_MODEL="")
    if asr == "sherpa":
        # Streaming Zipformer keeps producing partials for turn-taking and
        # barge-in; SenseVoice re-reads the finished sentence, which is the text
        # that reaches memory and the model. Measured offline on this machine:
        # 84-97 ms for a 5-second utterance, warmup 0.5 s.
        env["VOICE_ASR_SHERPA_MODEL"] = str(SHERPA)
        if SENSEVOICE.is_dir():
            env["VOICE_ASR_SENSEVOICE_MODEL"] = str(SENSEVOICE)
    elif asr == "vosk":
        env["VOICE_ASR_VOSK_MODEL"] = str(VOSK)
    elif asr == "whisper":
        env["VOICE_ASR_COMMAND"] = shlex.join([shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli",
                                              "-m", str(VOICE / "weights/ggml-base.bin"), "-f", "{wav}", "-l", "zh", "-nt", "-np", "-sns"])
        env["VOICE_ASR_VOSK_MODEL"] = str(VOSK)
    return env


def specs(session, asr="sherpa", camera=0):
    env = environment(session, asr)
    py = str(PYTHON)
    return {
        "qwen": Spec(["bash", str(MODELS / "scripts/run-qwen3.5-9b-llama-cpp.sh")], MODELS, env),
        "memory": Spec(["cargo", "run", "--", "--host", "127.0.0.1", "--port", "8788", "--data-dir", "./data"], WORKSPACE / "hri-memory-service", env),
        "vision": Spec([py, str(VISION / "scripts/local_vision_dashboard.py"), "--host", "127.0.0.1", "--port", "8080", "--camera-index", str(camera), "--emotion-backend", "ferplus", "--stream-fps", "20", "--inference-fps", "20"], VISION, env),
        "voice": Spec([py, str(VOICE / "scripts/local_voice_dashboard.py"), "--host", "127.0.0.1", "--port", "8090"], VOICE, env),
        "attention": Spec([py, "-m", "robot_attention_perception.live_dashboard", "--host", "127.0.0.1", "--port", "8092", "--vision-url", "http://127.0.0.1:8080", "--voice-url", "http://127.0.0.1:8090", "--memory-url", "http://127.0.0.1:8788", "--rosbridge-url", "ws://127.0.0.1:9090", "--session-id", session], ATTENTION, env),
        "brain": Spec([py, "-m", "brain.dashboard", "--host", "127.0.0.1", "--port", "8094"], ROOT, env),
    }


def required_files(names, asr="sherpa", tse="baseline"):
    files = {}
    if "qwen" in names:
        files["Qwen3.5-9B Q4_K_M"] = [QWEN, MODELS / "third_party/llama.cpp/build/bin/llama-server"]
    if "ros" in names:
        files["ROS audio_msgs contract"] = [WORKSPACE / "pacific-rim/module/service/robo_brain_service/ros2/src/audio_msgs/CMakeLists.txt"]
    if "vision" in names:
        for label, name in [("YOLOv8n", "yolov8n.onnx"), ("YuNet", "face_detection_yunet_2023mar.onnx"),
                            ("SFace", "face_recognition_sface_2021dec.onnx"), ("FER+", "emotion-ferplus-12-int8.onnx"),
                            ("MediaPipe Face", "face_landmarker.task"), ("MediaPipe Hand", "hand_landmarker.task")]:
            files[label] = [VISION / "models" / name]
    if "voice" in names:
        files["VAP turn prediction"] = [VOICE / "weights/vap/model.pt", VOICE / "vendor/voice_activity_projection/vap/model.py", VOICE / "vendor/voice_activity_projection/cpc_config.json"]
        if asr == "sherpa":
            files["Sherpa Zipformer Chinese（流式 partial）"] = [SHERPA / name for name in ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt")]
            files["SenseVoice（最终句，缺失则退回流式最终句）"] = [SENSEVOICE / "model.int8.onnx", SENSEVOICE / "tokens.txt"]
        if asr in {"whisper", "vosk"}:
            files["Vosk Chinese"] = [VOSK / "am/final.mdl", VOSK / "graph/HCLr.fst"]
        if asr == "whisper":
            files["Whisper base"] = [VOICE / "weights/ggml-base.bin", Path(shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli")]
        files["ERes2NetV2 声纹嵌入"] = [VOICE / "weights/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"]
        for label, folder in [("AST AudioSet music", "ast-audioset"), ("Wav2Vec2 ten genres", "music-genre")]:
            files[label] = [VOICE / "weights" / folder / name for name in ("model.safetensors", "config.json", "preprocessor_config.json")]
        if tse in {"speakerbeam","multi"}:
            files["REAL-TSE SpeakerBeam"] = [VOICE / "weights/real-tse/pretrained/spk_emb_causal_100" / name for name in ("avg_model.pt", "config.yaml")]
            if tse=="speakerbeam":files["Enrolled TSE reference"] = [VOICE / "config/speaker_references/target.wav"]
    if "attention" in names:
        # Missing plugins do not block startup: the console falls back to the
        # built-in audio-visual baseline and shows the load error.
        files["注意力算法 .so（缺失时降级为仅视听基线）"] = [ATTENTION / "plugins/libattention_av_memory_language_v1.so"]
    if "brain" in names:
        files["macOS speech synthesis"] = [Path("/usr/bin/say")]
    return files
