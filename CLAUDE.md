# biomimetic-brain-test：实现交接（2026-09-16）

## 工作规则与实际部署
- 用户授权跨六项目协作；大量未提交实现不可擅自清理或回滚。不创建子代理，除非用户明确要求。
- 修改前检查并停止本机受管服务，不擅自打开摄像头/麦克风或播放。远端162的Qwen为常驻服务，不能随本机stop_all误停。
- 机器人名称“小圆”。用户要求自然多人对话、旁人记忆、视听身份、注意力约束打断、主动搭话。声纹/视觉冲突优先可靠声纹归属发言，不据此改写冲突人脸。
- 本机启动：cd ~/Golands/biomimetic-brain-test && bash scripts/start_all.sh；停止：bash scripts/stop_all.sh；状态：python3 scripts/mac_stack.py status。
- Python：~/Golands/robot-attention-perception/.venv-mac/bin/python。MediaPipe固定0.10.32，不盲目升级。
- 端口：Vision8080、Voice8090、Attention8092、Brain8094、Memory8788、ROS9090；慢脑http://172.16.60.162:18050/v1/chat/completions，model=qwen3.5-9b。
- 所有基准和单元测试仅说明对应样本/程序行为，不能描述为现场识别准确率达标。
- 不在代码/文档保存SSH密码。完整历史见voice-detection/AGENTS.md下方历史记录；本节当前状态优先于过期历史。
- /Applications/声音/Mac全系统启动与三项验收.txt保持“一键启动／分别启动”两块，详细验收独立保存。

## 本项目已实现
- 当前唯一S2S决策所有者：消费Attention流（ROS优先、HTTP降级），保存所有合格听到的发言，门控后生成TTS。
- 远端Qwen3.5-9B SSE流式短句生成，Mac say合成，Voice播放与真实AEC参考；旧generation回调隔离、取消队列与模型socket、分级淡出/350ms短句让话上限。
- 当前问题+结构化近期16条/2分钟+召回历史最多12条+身份/背景/不确定性；无current_topic硬选最后句，千问自主理解转折/指代/讨论。current_turn.model_context保存实际输入。
- 温柔自然，简单回应3–9字，比较/总结允许1–2短句；提示词限制不是硬字数保证。默认max_tokens128。
- 语音注册“你好小圆，我叫小天，请记住我，我是你的主人”，两端预检后注册；重复主人请求俏皮确认；失败口头说明。
- 主人主动问候：必须锁定主人、连续安静5秒；开口重置，失锁取消，已交谈不补问候，短视觉hold不新发起问候。
- 群体问题700ms让话窗口，真人接话撤回待答。惊跳优先，新问题暂存；咳嗽确认后合适空闲时简短关怀，非健康诊断。
- 无声转向邀请与转头回答已上移到注意力层（见下方第二版）。Brain 侧只剩 attention_trigger.py：读 attention.engagement，判断自己是否空闲，按 transition_id 每个交流片段只开口一次；空 user_text 送千问按语境自然回应。
- dialogue_continuity.py 现为 ExpectedAnswer：只记录“机器人问完并已播完的问题、问的是谁、10秒有效期”，经 /brain/state 的 attention_memory 交给注意力。判定谁在回答由注意力做；放行后 Brain 仍把自己问过的原句附给千问（dialogue_link）。Brain 不再自己判断视线或“是的”。
- 当前端仍为MacTTS，不具备神经情绪音色。Qwen3-TTS-1.7B-CustomVoice官方Serena是温柔年轻中文女声候选（Vivian较明亮）；查证支持instruct情绪，尚未部署，不要误写已换声音。
- tts_service评估：pacific-rim该checkout缺tts/adapters/tts_engine.py，现有AudioFrame无明确轮次/取消/结束；不能直接替换。建议仅替换合成，继续Voice播放/AEC及Brain取消，IDL必须经Dashboard或./pr data-format，不能手改。

## 启动、验证与文件
- scripts/mac_stack*.py、llm_endpoint.py：默认162远端健康检查，all跳过本机Qwen；本机stop_all不触碰远端。run_brain.sh默认远端；BRAIN_LLM_BASE_URL可覆盖。
- brain/{runtime,llm,prompt_context,memory,attention_trigger,dialogue_continuity,streaming_speech,ros_stream,io}.py。gaze_invitation.py 已停用未删除，运行时不再引用，确认后可自行删除（连同 tests/test_gaze_invitation.py）。
- 本轮完整66项测试通过，包含注意力交流意愿触发、预期回答记忆事实与上下文验证。页面DOM通过；两条真实Qwen非播放接话样本验证，非现场准确率。
- data/outbox.sqlite3不可删除；docs与/Applications/声音包含启动、六项准确率计算、部署、仿生打断、多人原型、TTS兼容说明。

详细历史与最新追加记录请同时阅读本项目AGENTS.md，冲突以用户最新指令为准。

## 统一注意力竞争（第五版，2026-09-16 最新）
理论链已全部落地：Itti-Koch 显著性 → Biased Competition → Reynolds-Heeger 除性归一化（v2，归一化池跨模态耦合 κ=0.35）
→ Selective Tuning（v2 方位抑制环 + workspace soft-WTA 压制非赢家）→ Habituation/IOR/Hysteresis → Global Workspace → Brain。
唯一没做的是层级 Selective Tuning：候选空间是扁平的 person/track/object，没有真正的 part-whole 层级，硬套是空架子。
输入源五个（8092 可勾选，存 .run/attention-config.json）：audio_visual、memory_context、linguistic_context、
cross_session_memory、internal_state（默认关闭，无生产者）。关闭=消融该通道证据，不是关硬件或安全反射。
算法为独立 .so（编译 bash scripts/build_attention_plugins.sh）：av_memory_language_v1（默认，自上而下相加）、
av_memory_language_v2（乘性注意场 + 跨模态池 + 抑制环）、builtin_audio_visual_v1（仅视听基线）。
v1/v2 跑同一套行为用例全过；现场优劣未比较，所以默认仍是 v1。缺 .so 不阻断启动，回退基线并报错。
Global Workspace（global_workspace.py）是统一订阅点也是认知层竞争：soft-WTA(share=a²/Σa²) + 有限容量 4 +
urgency≥0.7 抢占；非感知候选（记忆、惊跳）在这里竞争，因为它们没有模态。
入口 GET /api/workspace、?since=N 回放、/api/workspace/stream (SSE)、ROS /attention/workspace。订阅者落后丢帧并计数。
双向：Memory→Attention 用 memory_candidates.py（保守：只提未完成的事、每 5 分钟一次、人在场、安静 3 秒、
同一件事只提一次，ATTENTION_MEMORY_EVENTS=0 可关）；Brain→Attention 用 attention_memory.goals 作 top-down bias
（awaiting_answer/holding_floor/deferred_turn/greet_owner，只偏置不放行）。
记忆通道整体乘以识别置信度；角色（主人/陌生人）不进交流意愿层，只留在第一层 importance 影响「看哪里」。
模型：声纹 ERes2NetV2（192 维）；人脸走商用路线，默认 sface(Apache-2.0) + 质量门槛 + 5 帧模板均值，
arcface(buffalo_l) 已下载但权重仅限非商业研究，--face-backend arcface 可切；身份识别 300ms 节流。
Attention 164、Brain 62、Vision 23、C++ 4 个测试目标、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查。
权重与阈值是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第四版历史，2026-09-16）
输入源五个（8092 可勾选，存 .run/attention-config.json）：audio_visual、memory_context、linguistic_context、
cross_session_memory、internal_state（默认关闭，无生产者）。关闭=消融该通道证据，不是关硬件或安全反射。
算法为独立 .so（C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh）：
av_memory_language_v1（默认，自上而下相加）、av_memory_language_v2（Reynolds-Heeger 乘性注意场 + Selective Tuning
方位抑制环，交流意愿层与 v1 共享 attention_common.hpp）、builtin_audio_visual_v1（内置仅视听基线）。
v1/v2 跑同一套行为用例且全过；两者优劣未经现场对比，所以默认仍是 v1。缺 .so 不阻断启动，回退基线并报错。
统一订阅点 global_workspace.py：每个感知周期广播一份内容（cycle/focus/target/engagement/coalition/sources/
algorithm/provenance）。入口 GET /api/workspace、/api/workspace?since=N 回放、/api/workspace/stream (SSE)、
ROS /attention/workspace。订阅者落后会丢帧并计数，不阻塞感知循环。Brain 现有 brain_input 通路未改。
结构：自下而上（视听显著度、声学突变、句首唤醒词）+ 自上而下（工作记忆目标、熟悉度先验）的归一化竞争，
其上再加一层按人的序贯证据累积决定「是不是在跟我交流」。内部状态将来接 arousal/motivation 与第二层通道。
记忆通道整体乘以识别置信度；角色（主人/陌生人）不进交流意愿层，只留在第一层 importance 影响「看哪里」。
模型已替换：声纹 CampPlus -> ERes2NetV2（192 维，3 秒语音 14.1->53.9ms，句末计算）；
人脸新增可切换后端，默认仍是 sface（Apache-2.0），arcface(buffalo_l w600k_r50) 已下载但权重仅限非商业研究，
需确认用途后用 --face-backend arcface 启用；身份识别加 300ms 节流。旧声纹/人脸档案已按指示清空，
备份在 ~/Golands/.identity-backup-*。阈值未在真人数据上标定。
Attention 151、Brain 62、Vision 16、C++ 4 个测试目标、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查（九个场景，含三组消融/对照）。
权重与阈值是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第三版历史，2026-09-16）
注意力输入源与算法都已解耦，8092 面板「注意力输入源与算法」可勾选来源、切换算法，选择存 .run/attention-config.json。
来源五个：audio_visual、memory_context（本会话工作记忆）、linguistic_context（当前语境文本）、
cross_session_memory（跨会话熟悉度，后台查 hri-memory-service，快循环只读缓存）、internal_state（默认关闭，无生产者）。
算法为独立 .so，C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh：
av_memory_language_v1（默认，视听+记忆+语境，含按人累积的交流意愿层）、builtin_audio_visual_v1（内置仅视听对照基线）。
缺少 .so 不阻断启动，自动回退基线并在面板显示加载错误与「无交流意愿层」。
结构上是 自下而上（视听显著度、声学突变、句首唤醒词）+ 自上而下（工作记忆目标、熟悉度先验）的归一化竞争，
其上再加一层按人的序贯证据累积决定「是不是在跟我交流」。内部状态将来作为 arousal/motivation 接第一层、连续通道接第二层。
记忆通道整体乘以识别置信度：身份不可靠时那份历史不能替当前这个人说话；角色（主人/陌生人）不进入交流意愿层，
只保留在第一层 importance 影响「看哪里」，权限与内容仍归 Brain。
无声邀请、转头回答、自言自语后转向、称呼他人、熟人回访这些情况不再有各自的代码分支，由同一次融合更新产生。
Attention 143 项、Brain 62 项、C++ attention_plugin_test 9 项、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查（九个场景，含三组消融/对照）。
权重是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第二版历史，2026-09-16）
注意力输入源与算法都已解耦，8092 面板"注意力输入源与算法"可勾选来源、切换算法，选择存 .run/attention-config.json。
来源：audio_visual、memory_context、linguistic_context（当前语境文本，新增）、internal_state（默认关闭，无生产者）。
算法为独立 .so，C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh：
av_memory_language_v1（默认，视听+记忆+语境，含按人累积的交流意愿层）、builtin_audio_visual_v1（内置仅视听对照基线）。
缺少 .so 不阻断启动，自动回退基线并在面板显示加载错误与"无交流意愿层"。
无声邀请、转头回答、自言自语后转向、称呼他人这些情况不再有各自的代码分支，由同一次融合更新产生。
Attention 132 项、Brain 66 项、C++ attention_plugin_test、页面 DOM 通过；scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查。
权重是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力输入（第一版历史，2026-09-16）
Brain snapshot新增attention_memory（近期参与者/最近发言/预期回答对象与有效期），既有/brain/state传给Attention。长期语义历史仍检索供千问，快速注意循环只读有限工作记忆，不反复请求LLM。Brain60项通过。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。没有内部状态也可试用第一版，但现场可靠性未验收，不能搬用论文准确率。

## 会话切换交接（2026-09-16，最新）
新AI先读 `/Users/letianxing/Golands/voice-detection/SESSION_HANDOFF.md`，再读本项目当前摘要。
本轮已核查：本机全部感知/Brain服务停止，162远端Qwen健康运行。最新Attention101/Brain60/DOM测试通过，不等于现场准确率。没有提交git，保护所有未提交修改和用户数据。
