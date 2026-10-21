# biomimetic-brain-test

Mac 多人对话测试大脑：独立监听 Attention；所有最终转写写入 HRI 记忆；只有注意力确认面向机器人的话语才触发回复或打断。主人出现时结合其记忆、视觉情绪及本地 Qwen 主动搭话。动作仅记录意图。

## 启动

```bash
cd ~/Golands/biomimetic-brain-test
bash scripts/start_all.sh
```

包含Qwen、ROS2、Memory、Vision、Voice、Attention、Brain，自动开始本机摄像头/麦克风采集并打开面板。启动前检查模型文件与Python依赖。使用MicArray可加 `--profile sipeed_6_plus_1_usb_array --device MicArray`。

分开启动：`bash scripts/start_component.sh qwen|ros|memory|vision|voice|attention|brain`（选择其中一个，不要照抄竖线）。共用同一套配置；默认Sherpa，可用 `voice --asr whisper` 或 `voice --asr vosk` 切换。

```bash
bash scripts/stop_all.sh
python3 scripts/mac_stack.py check
python3 scripts/check_mac_conversation.py
```

启动文档仅分一键/分别启动两块：`docs/Mac全系统启动与三项验收.txt`；详细注册和验收见 `docs/现场注册与验收.txt`。两份均复制到 `/Applications/声音/`。旧 `run_five_services.sh` 保留兼容，不要和新启动器同时运行。

## 打断与记忆

- Brain 用 `say` 合成 PCM，经 Voice 播放；Voice 从同一输出回调提供时间戳回声参考，并在输入端估计 0–240ms 声学延迟后做轻量回声抵消。播放时持续识别。
- Attention 的 `interruption_candidate` 必须有新鲜的流式转写、有效人声和指向机器人的证据；Brain 连续确认 120ms 后停止 PCM，取消 HTTP LLM 请求并使旧回复失效。120ms 是确认窗，不是端到端打断延迟。
- 定向称呼采用句首“小圆/机器人/Reachy”等；视觉指向需要同一声源关联的人脸注视及唇动/朝向证据。第三方句中提到“机器人”不会单独唤醒。
- 每句 ASR 附带当句声纹、时间和重叠信息，Attention 用说话当时的视觉/声源证据标注。`utterances` 是对话门控证据；旧面板的 `attention` 仍是实时融合状态，两者可能因时序不同而不同。
- 不对机器人说话也保存 `heard_utterance`，包含 person_id、身份来源/可信度、addressee、可见参与者。未知身份和未知对话对象明确保留，不能把“可见参与者”解释为确定听话人。
- Brain SQLite outbox 持久化、重试；HRI 对相同 trace/source/kind/内容的重试去重。检索读取 `hits[].record`，结合中文片段匹配和最近对话。固定 SESSION_ID 才会检索该会话记忆。
- 主人需通过视觉身份注册，稳定出现 800ms、环境语音空闲 2s 后搭话；离开至少 3s 后再次出现才重新触发。Qwen 读取主人记忆和情绪线索，持续在场不定时重复问候。

## 可验证边界

这是可测试的双向回话实现，不是现场准确率保证。当前声学前端每块输出主要声源：多人轮流发言可用登记声纹区分；单麦多人同时重叠说话尚不能保证各人的完整独立转写。多人记忆测试用 Baseline，SpeakerBeam 目标提取模式不适合完整记录所有旁人对话。回声抵消是轻量实现，外放音量、房间混响、ASR 和人物关联都会影响真实打断延迟与误触发率。

`hri-memory-service` 是 JSONL 事件存储、作用域过滤和文本/可选向量检索服务；不能仅据此宣称与未提供的 `memory_v2_service` 完全等价。

## 自动测试

```bash
python3 -m unittest discover -s tests
```

覆盖旁人只记忆、注意力确认打断、过期 LLM 回复丢弃、主人出现/离开、记忆 outbox 重启及 HRI 返回协议。Voice 和 Attention 项目另有回声参考对齐、播放停止及时间绑定测试。
