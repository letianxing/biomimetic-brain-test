# biomimetic-brain-test：当前实现与交接（2026-09-16）

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
- 2026-09-16新增：单人自言自语/多人讨论后的无声转向邀请。最近90秒同episode有真实发言且目标属于参与者/可见者，持续VISUAL_FOCUS1.5s+安静1s，无busy/pending后产生attention_trigger，空user_text送Qwen自然回应，同episode去重。
- DialogueContinuity：机器人确已播放完的问题保留10s预期回答；同一可靠声纹、非回声/非混合、未明确指向别人时，转头回答也可延续。保留observed_attention原值并用dialogue_link说明放行依据；不是把视线或“是的”当唯一规则。
- 当前端仍为MacTTS，不具备神经情绪音色。Qwen3-TTS-1.7B-CustomVoice官方Serena是温柔年轻中文女声候选（Vivian较明亮）；查证支持instruct情绪，尚未部署，不要误写已换声音。
- tts_service评估：pacific-rim该checkout缺tts/adapters/tts_engine.py，现有AudioFrame无明确轮次/取消/结束；不能直接替换。建议仅替换合成，继续Voice播放/AEC及Brain取消，IDL必须经Dashboard或./pr data-format，不能手改。

## 启动、验证与文件
- scripts/mac_stack*.py、llm_endpoint.py：默认162远端健康检查，all跳过本机Qwen；本机stop_all不触碰远端。run_brain.sh默认远端；BRAIN_LLM_BASE_URL可覆盖。
- brain/{runtime,llm,prompt_context,memory,gaze_invitation,dialogue_continuity,streaming_speech,ros_stream,io}.py。
- 本轮完整60项测试通过，包含单人转向、视线转开后的同人问答连续性与上下文验证。页面DOM通过；两条真实Qwen非播放接话样本验证，非现场准确率。
- data/outbox.sqlite3不可删除；docs与/Applications/声音包含启动、六项准确率计算、部署、仿生打断、多人原型、TTS兼容说明。

## 可插拔注意力输入（2026-09-16）
Brain snapshot新增attention_memory（近期参与者/最近发言/预期回答对象与有效期），既有/brain/state传给Attention。长期语义历史仍检索供千问，快速注意循环只读有限工作记忆，不反复请求LLM。Brain60项通过。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。没有内部状态也可试用第一版，但现场可靠性未验收，不能搬用论文准确率。

## 会话切换交接（2026-09-16，最新）
新AI先读 `/Users/letianxing/Golands/voice-detection/SESSION_HANDOFF.md`，再读本项目当前摘要。
本轮已核查：本机全部感知/Brain服务停止，162远端Qwen健康运行。最新Attention101/Brain60/DOM测试通过，不等于现场准确率。没有提交git，保护所有未提交修改和用户数据。

## LLM 上下文被撑爆导致整轮回复消失 + 上下文结构化（2026-09-16，最新）
现场「TTS 回复时 LLM 报错」的真因，来自 brain_trace：
  LLM HTTP 400: request (4622 tokens) exceeds the available context size (4096 tokens)
  LLM HTTP 400: request (5901 tokens) ...
远端 llama-server（172.16.60.162:18050，Qwen3.5-9B Q4_K_M）slot n_ctx=4096，单 slot。
超长不是「答得差」，是服务端整个请求拒收、这一轮什么都没有——而用户那边已经在等回话了。

两件事一起做：
1) 硬预算。prompt_context.fit_to_budget()：先算保守 token 数（对 Qwen3.5 实测中文约 0.6 token/字、
   JSON 骨架约 0.3，代码按 0.8/0.4 算，宁可早裁），超预算就按顺序丢——先丢 history（召回来的，
   现场对话才是要接住的），再丢最旧的 recent，最后清 preceding_conversation；当前这句和正在回答的问题永不丢。
   还塞不下就把过长的发言截短（120/60/30 字），保住「谁说了什么」。
   丢了什么写进 context_limits.dropped_for_context_size，模型和 trace 都不会把残余当成全部。
   预算 = BRAIN_LLM_CONTEXT_TOKENS(4096) − BRAIN_MAX_TOKENS(128) − 256 余量。
   llm.py 另加一次重试：真收到 exceed_context_size 就用一半预算重发，短一点总比这轮死掉好。
2) 结构化（用户要求）。原来每条发言都带两三个 13 位毫秒时间戳和完整 session id，
   一个就是五到七个 token，模型还读不出意义；改成相对当前发言的秒数 t（负数为更早），
   跨会话只留 earlier_session 标记，identity 只在有冲突时出现，visual_observations 只保留最近两条发言的，
   current_turn 的毫秒戳换成「时:分」（问候和「下午三点」要用到，毫秒用不到）。
   用现场 events.jsonl 的真实对话、同样 12 条 history + 16 条 recent，
   用远端自己的 /tokenize 量：3201 → 2312 token，省 28%，且没有触发任何裁剪。
   估算值 2343 对实际 2312，保守但不虚高。
不动的：n_ctx 仍是 4096。要不要把远端 slot 调大是用户决定（要改 systemd robo-qwen9.service 并重启远端模型），
本机侧现在无论如何都不会再发出超限请求。
新增用例 ContextSizeTests 三条（超限会裁并说明裁了什么、history 先于现场对话让位、毫秒戳与 session id 不再出现）。
机器人名字改为「小圆」（含同音写法 小园/小元），SYSTEM_PROMPT、registration、core 的唤醒前缀同步改。
验证：brain 75、voice 82、attention 182、vision 27 全过；远端只做了只读 /tokenize 与 /props 查询，没重启。

## 回复质量、工具、主动性、打断（2026-09-16，最新）
用户现场反馈四件事：很容易被打断；回复质量低、几乎都是反问；盯着它很久也不主动开口；
多人聊天后转头问「我刚说的对吗」答不到点上。逐条处理，能量的都量了。

1) 提示词重写（brain/prompt_context.py）。原提示要求「3～9字为主」「不确定就说明」，
   在没有工具的情况下，模型唯一安全的出口就是反问，于是现场几乎每句都是「你是问…吗」。
   现在：先给实质内容或判断，禁止反问开场；只有关键信息在 recent/history 里都找不到、且猜错会误导时才问一句；
   一般一到两句，简单确认仍可三到九个字，比较/总结多人观点最多三句。
   新增一段专门写多人场景：recent 里是旁人之间的对话，有人转头问「我刚才说的对吗」时，
   要回到 recent 里那个人最后几句的具体内容给判断（同意/不同意/补依据），不复述、不反问。
   无声邀请那条也改了：原来写「语境不够明确就温柔询问是否需要你的意见」，这正是「需要我帮忙吗」的来源。
2) 大脑变成 agent（brain/tools.py + llm.py 的工具循环）。远端 llama-server 的 chat_template_caps
   确认支持 tools/tool_calls，流式下 tool_calls 也是分片到达的，已按分片拼接。
   两个工具：get_weather（open-meteo，无需 key，地理编码+实况+今日高低温，城市取 BRAIN_LOCATION）、
   get_datetime（模型没有时钟）。一轮工具调用上限，工具调用不产出朗读文本，所以不会先说半句再改口。
   工具失败返回 error 让模型说「没查到」，不编。地理编码结果常驻缓存、天气缓存 10 分钟（BRAIN_WEATHER_TTL）。
   BRAIN_LOCATION 默认空：没人告诉机器人它在哪，就回答不知道，而不是报另一个城市的天气。
3) 主动性：注意力侧新增「持续对视」证据（见 robot-attention-perception/AGENTS.md）。
4) 打断：turn_taking 的确认窗口从 120/200ms 提到 200/350ms；视觉打断（嘴动+注视、还没有词）
   需要连续 400ms 且麦克风没在收自己的回声。显式「停一下」「小圆」「小心」仍然是短窗口。
新增 scripts/verify_reply_quality.py：把现场答错的那几种情况写成固定输入，走同一套 build_context 和
同一个 LLM 客户端，打真实远端模型，打印它到底说了什么 + 几条粗检查（有没有反问开场、有没有用对说话人的内容、
天气答案里有没有数字）。不开麦克风/摄像头/喇叭。实测（BRAIN_LOCATION=深圳）：
  「你说我刚刚说的对吗」→「对，口径不一致确实得先跟对方对齐，不然现场容易吵起来。」733ms
  「你觉得呢」→「二丫说得对，口径不一致确实得先跟对方对齐…」557ms
  无声邀请 →「天行，数据核对和口径对齐确实得先搞定…」569ms（不再是「需要我帮忙吗」）
  「今天天气怎么样」→ 调 get_weather →「今天深圳多云，气温26.7度，最高31.1度，最低24.6度。」冷缓存 3.7s
这几条只说明这几句文本符合我们写下的要求，不是对话质量评分，更不是现场准确率。
「二丫你觉得这样行吗」这类不该进大脑的句子由注意力门控负责，在 attention 的离线九场景里验证，不在这个脚本里。
验证：brain 77、voice 84、attention 183、vision 27 全过；远端只做了只读查询和一次按用户要求的重启。
