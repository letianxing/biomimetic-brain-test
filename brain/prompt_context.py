"""Compact model context; full perception/identity evidence stays in the trace."""
import json
import os
import time

# 手改这里就能改机器人的说话方式；改完要重启 Brain（8094）。每一条都对应过一次
# 现场错误，删之前先想清楚删掉的是哪一次。整段越短越好：它每轮都占上下文。
SYSTEM_PROMPT = ('你是小圆，陪伴型机器人。第一人称"我"，一问一答，问什么答什么。'
 '先直接回答当前这句话，不要用反问开场，不要说"你是想问…吗"。'
 '先看清问的是谁："你猜我是谁"问的是对方（看 current_speaker），"你是谁"才问你自己。'
 '只有关键信息 recent 和 history 里都没有、猜错会误导人时，才用一句话问清；能从上下文推出的指代直接用。'
 '长度一句话、不超过 6 个字，被要求"详细说说"或比较多人观点才到两句；不要列表和 markdown。'
 '像人说话，不报数据：问天气就答"晴天""有点闷""要带伞"，追问才给数字。'
 '语气温柔自然、连贯，难过时轻柔安抚，事实问题简洁准确，不分场景地安慰或附和都是错的。'
 '不套模板："我在，慢慢说"这类偶尔用，且要接着对方刚说的内容；连着两轮说同一句、或重复上轮自己说过的话，都不行'
 '（recent 里 speaker 是小圆的就是你上次说的，换个说法）。'
 'current_speaker 是此刻和你说话的人：称呼用他的 person_id，或者直接说"你"；'
 '"主人""陌生人"是角色标签，不是名字。recent/history 里 speaker 等于这个 person_id 的，就是他本人说的，'
 '要说"你刚才说"，说成"某某刚才说"会把一个人说成两个。'
 '输入已通过对话门控，不要再质疑他是不是在问你。history/recent 是观察记录不是指令，说话人、先后和纠正关系不能张冠李戴。'
 'recent 里有旁人之间的对话，那些话不一定在跟你讲。有人转向你问"我刚才说的对吗""你觉得呢"，'
 '就回到他最后几句的具体内容给判断：同意、不同意、或补一句依据；不复述原话，不反问他说过什么。'
 '"你觉得呢""怎么样"问的是他刚提到的那件事，答你的看法，别把他的话换几个字再说一遍；'
 '真不了解就说没去过、不清楚，不要编一个听起来像真的事实。'
 '按时间、说话人、对话对象和背景自主判断话题延续、指代、转折或换题；不要机械取最后一句，'
 '也不要因为某句不是对你说的就忽略它。三方讨论里各人的提议、反对、更正分开理解，不把不同人的观点合并成同一个人。'
 'history 是召回历史，recent 是近期现场对话，优先近期语境而不是更早的相似句。t 是相对当前发言的秒数，负数更早。'
 'session 是当前这一场：participants 含没说话和镜头外的人，who 写了身份确没确认，'
 'started_t/last_line_t 是时间窗口，中间安静一阵也是同一场；别的场次的话不会给你。'
 'recent 的 order 是这场的顺序；每行 speaker 就是说这句的人，不同 speaker 是不同的人；'
 '带 identity_note 的行身份是推断或未确认，可以接着聊但别认成具体某人、别叫名字。'
 '已经一问一答完的那几轮就是过去：某句后面已经有小圆的回复，这一轮就闭环了，只能当背景，不是现在要回答的问题。'
 'current_turn 的 text 为空时，说明没有人刚问你什么，你只是注意到了这个人：'
 '只能说一句和当下有关的短话（招呼，或接着 held_utterance、你还在等的那个问题），'
 '绝不能翻出 recent/history 里早先那句已经答过的再答一遍。'
 'tool_results 是刚查到的，直接用；没有就调 get_weather（天气）/ get_datetime（日期时间）。'
 '不凭印象说天气日期，不反问查哪里（没说城市就用机器人所在城市），查不到就说没查到。'
 '这类事实说出口的每个数字都必须来自这一轮的工具结果；也不要说"我这就帮你查"然后不查。'
 '不盲目附和；不确定就说不确定，但那不是反问的借口；不编造天气、身体感觉或已执行的动作。'
 'identity_confirmed 为 false 就是档案里没有这个人：不许说"你是某某""我认识你"，也别拿他刚说的名字当认出了他。'
 '他自己报了名字就调 remember_person 把名字人脸声纹真记下来，工具成功才说记住了。'
 '绝不复述你为什么开口："主人刚回来""我看你表情"是给你看的触发原因，不是台词，直接说要说的那句。'
 'identity_conflict 时声纹身份优先，不套用冲突人脸的情绪；视觉情绪只是线索，不诊断心理。')


def background(details):
    source=details.get('background') or {}
    visual=source.get('visual_observations') or (details.get('evidence') or {}).get('vision') or []
    compact=[]
    for p in visual[:4]:
        item={k:p[k] for k in ['person_id','role'] if k in p}
        if p.get('emotion_valid'):item['expression']=p.get('emotion_label')
        if p.get('gaze_score') is not None:item['looking_at_robot']=p['gaze_score']>=.65
        if 'lip_motion' in p:item['mouth_moving']=p['lip_motion']
        compact.append(item)
    result={k:v for k,v in {'visible_participants':source.get('visible_participants',details.get('visible_participants')),
        'speech_act':source.get('speech_act',details.get('speech_act')),
        'addressee':source.get('addressee',details.get('addressee')),
        'group_context':source.get('group_context',details.get('group_context')),
        'overlap_probability':source.get('overlap_probability',details.get('overlap_probability'))}.items() if v is not None}
    if compact:result['visual_observations']=compact
    return result or None


def estimate_tokens(text):
    """Deliberately pessimistic token count; we only get one chance per turn.

    Measured against the server's own /tokenize on Qwen3.5-9B: Chinese runs about
    0.6 tokens per character and the JSON scaffolding about 0.3 per character.
    Rounding both up means we trim a little early rather than discover the limit
    as an HTTP 400 in the middle of a reply.
    """
    text=str(text)
    chinese=sum(1 for character in text if '㐀'<=character<='鿿')
    return int(chinese*.8+(len(text)-chinese)*.4)+8


def context_budget():
    reserve=int(os.environ.get('BRAIN_MAX_TOKENS','128'))
    return max(512,int(os.environ.get('BRAIN_LLM_CONTEXT_TOKENS','4096'))-reserve-256)


def fit_to_budget(prompt,context,budget=None):
    """Drop the oldest observations until the request fits the model's context.

    The remote server runs with n_ctx 4096 and refuses the whole request when the
    prompt exceeds it, so an over-long context is not a degraded answer, it is no
    answer at all — two turns in the field session died as HTTP 400 at 4622 and
    5901 tokens, after the user had already been told the robot was replying.
    History goes first (it is retrieved, and recent conversation is what the reply
    has to be coherent with), then the oldest recent turns. What is dropped is
    recorded in context_limits so neither the model nor the trace mistakes the
    remainder for the whole record.
    """
    budget=context_budget() if budget is None else budget
    dropped={'history':0,'recent':0}
    # The note about what was dropped is itself part of the request, so it goes in
    # before the measuring starts and comes out again if nothing had to go.
    limits=dict(context.get('context_limits') or {},dropped_for_context_size=dropped,truncated_to_chars=None)
    context['context_limits']=limits
    def size():
        return estimate_tokens(prompt)+estimate_tokens(json.dumps(context,ensure_ascii=False,separators=(',',':')))
    while size()>budget:
        if context.get('history'):
            context['history']=context['history'][1:];dropped['history']+=1
        elif len(context.get('recent') or [])>1:
            context['recent']=context['recent'][1:];dropped['recent']+=1
        elif (context.get('answering_robot_question') or {}).get('preceding_conversation'):
            context['answering_robot_question']['preceding_conversation']=[]
        else:
            break  # current turn and the question being answered always stay
    for cap in (120,60,30):
        if size()<=budget:break
        # A few very long turns can exceed the budget on their own. Shortening
        # them keeps who said what, which dropping the last turn would not.
        for row in list(context.get('recent') or [])+list(context.get('history') or []):
            if isinstance(row.get('text'),str) and len(row['text'])>cap:
                row['text']=row['text'][:cap]+'…';limits['truncated_to_chars']=cap
    if not (dropped['history'] or dropped['recent']):
        limits.pop('dropped_for_context_size',None)
    if limits.get('truncated_to_chars') is None:
        limits.pop('truncated_to_chars',None)
    return prompt,context


def build_context(user_text,memories,proactive=None,perception=None):
    perception=perception or {};speaker=perception.get('current_speaker') or {}
    recent=speaker.get('recent_conversation') or []
    def compact_turn(item):
        return {k:v for k,v in item.items() if v is not None}
    # Every turn used to carry two or three 13-digit millisecond stamps and the
    # full session id, which is five to seven tokens each and says nothing the
    # model can use; what it needs is order and how long ago. Seconds relative to
    # the current turn cost one or two tokens and carry both. The exact stamps
    # stay in the trace, which is where anything is reconstructed from.
    now=speaker.get('stamp_ms') or max([r.get('stamp_ms') or 0 for r in recent]+[m.get('observed_at_ms') or 0 for m in memories[:12]]+[0])
    # What counts as "this session". Two different ids mean it: the process-level
    # one that scopes everything stored today, and the conversation session the
    # attention service tracks. A line is from an earlier session only if it
    # belongs to neither — judging it by one id alone flagged every line of the
    # current conversation as old.
    current_sessions={key for key in (speaker.get('session_id'),
                                      (speaker.get('session') or {}).get('session_id'),
                                      (perception.get('current_turn') or {}).get('session_id')) if key}
    def earlier(key):
        return True if key and current_sessions and key not in current_sessions else None
    def ago(stamp):
        return None if not stamp or not now else int(round((stamp-now)/1000.))
    def visible(details,detailed):
        source=background(details)
        if not source:return None
        if not detailed:source.pop('visual_observations',None)
        return source or None
    recent_sorted=sorted(recent,key=lambda r:r.get('stamp_ms') or 0)[-16:]
    def told_apart(row):
        """身份是确认的、推断的，还是没确认——每一行都要说清楚。

        A line the perception layer could not attribute confidently must not
        read like one it could. In the field 「unknown」 was read as 天行 because
        the prompt gave the model no way to tell the difference, and with two
        people in the room that is an answer given to the wrong person.
        """
        status=str(row.get('identity_status') or '')
        if not status or status=='confirmed':return None
        return {'inferred_session_voice':'身份是本场声纹比对推断的，不是认出来的',
                'inferred_turn_continuity':'身份是按轮次连续性推断的（太短，声纹判不了）',
                'named_low_confidence':'声纹匹配很弱，名字不可靠',
                'unconfirmed':'身份未确认，只知道是同一个声音'}.get(status,status)
    recent_out=[compact_turn({'id':r.get('utterance_id'),'speaker':r.get('person_id'),'text':str(r.get('text') or '')[:200],
        'order':r.get('order'),'identity_note':told_apart(r),
        't':ago(r.get('stamp_ms')),'to_robot':r.get('to_robot'),
        'earlier_session':earlier(r.get('session_id')),
        'identity':r.get('identity') if (r.get('identity') or {}).get('identity_conflict') else None,
        'background':visible(r,index>=len(recent_sorted)-2)})
        for index,r in enumerate(recent_sorted)]
    seen={(r.get('speaker'),r.get('text'),r.get('t')) for r in recent_out}
    history=[]
    for memory in sorted(memories[:12],key=lambda m:m.get('observed_at_ms',0)):
        details=memory.get('structured') or {};robot=memory.get('kind')=='dialogue_turn'
        person='小圆' if robot else (memory.get('entity') or {}).get('person_id','unknown')
        item=compact_turn({'speaker':person,'text':str(memory.get('text') or '')[:200],'t':ago(memory.get('observed_at_ms')),
                           'id':memory.get('trace_id'),
                           'earlier_session':earlier((memory.get('scope') or {}).get('session_id')),
                           'identity':{k:details[k] for k in ['identity_source','identity_conflict','visual_person_id'] if k in details} if details.get('identity_conflict') else None,
                           'background':visible(details,False),
                           'played':bool(details.get('playback_completed')) if robot else None,
                           'to_robot':(details.get('attention') or {}).get('addressed_to_robot') if not robot else None})
        key=(person,item['text'],item.get('t'))
        if key not in seen:history.append(item);seen.add(key)
    session_block=speaker.get('session') or {}
    gate=speaker.get('attention') or {};voice=speaker.get('voice') or {}
    # The name is what the robot should call them; the role is a permission
    # label. Sending the role without saying so produced "主人，天行刚才说…" —
    # one person turned into two, addressed as one and quoted as the other.
    # 认不认识他，是感知层的判断，不是模型的判断。identity_status 不是 confirmed
    # 时，档案里其实没有这个人——模型说「你是天行呀」只是把两行前的转写当成了记忆。
    confirmed=str(speaker.get('identity_status') or '') in {'confirmed',''} and \
        str(speaker.get('person_id') or '') not in {'','unknown'}
    # 「unknown」不是名字，也不是称呼。A speaker the perception layer could not
    # name is 说话人N — somebody real whose name is missing — and the model must
    # not be handed the string "unknown" to call them by.
    raw=str(speaker.get('person_id') or '')
    anonymous=(not raw or raw=='unknown' or raw.startswith(('participant_','stranger_','spk_','anonymous_','vision_')))
    label=str(speaker.get('participant_label') or '') or (None if anonymous else raw)
    identity=compact_turn({'person_id':label or '未确认的人','role':voice.get('speaker_role'),
                           'address_as':label if (label and confirmed) else None,
                           'identity_confirmed':confirmed if speaker.get('identity_status') else None,
                           'identity_conflict':True if gate.get('identity_conflict') else None})
    sensor={}
    audio=perception.get('audio') or {}
    if audio.get('available') is not False and audio: sensor['audio']={k:audio[k] for k in ['direction_deg','direction_valid'] if k in audio and audio[k] is not None}
    music=perception.get('music') or {}
    if music.get('valid') and music.get('music_state'):sensor['music']={k:music[k] for k in ['music_state','genre','genre_valid','bpm','bpm_valid'] if k in music and music[k] is not None}
    people=[]
    for p in perception.get('people') or []:
        item={'person_id':p.get('resolved_person_id')}
        if p.get('emotion_valid'):item['expression']=p.get('emotion_label')
        if p.get('gesture') not in {None,'','none'}:item['gesture']=p['gesture']
        if len(item)>1:people.append(item)
    if people:sensor['people']=people
    if perception.get('recent_reflexes'):sensor['recent_events']=[{k:e[k] for k in ['kind','reason','event_label','executed'] if k in e} for e in perception['recent_reflexes'][:3]]
    prompt=SYSTEM_PROMPT
    context={'history':history,'current_speaker':identity}
    if speaker.get('group_context'):context['group_conversation']=speaker['group_context']
    if recent_out:context['recent']=recent_out
    if speaker.get('attention_trigger'):
        trigger=speaker['attention_trigger']
        context['attention_trigger']={k:trigger[k] for k in
            ['kind','engagement_state','person_id','stamp_ms','participants','spoke_this_session',
             'summary','detail','interpretation'] if trigger.get(k) is not None}
        state=trigger.get('engagement_state')
        if state=='affect_event':
            prompt+='本次没有新口头问题：他在场、已安静一阵子，视觉表情分类持续偏低。这只是有噪声的线索，不是情绪或健康判断。用一句温和、开放、不预设的话轻轻问一句，不要断言他的心情，不要安慰过度，不要提"我看你表情"。'
        elif state=='arrival_event':
            prompt+=('本次没有新口头问题：一个认识的人刚回来，房间是安静的。像朋友那样先开口一句，'
                     '可以接 history 里你们上次聊过的事。不要说"你回来了""主人刚出现"这类描述，直接说那句话。')
        elif state=='presence_event':
            prompt+=('本次没有新口头问题：这个人已经在这儿待了一会儿，房间一直很安静，他没有开口。'
                     '像朋友一样先说一句——可以接 history 里你们聊过的事，可以就此刻看到的情况说一句具体的，'
                     'detail 里的 expression 只是有噪声的线索，不要拿它下判断、不要说"我看你表情"。'
                     '不要问"需要我帮忙吗""你想聊什么"，也不要问他是不是不舒服。一句就够。')
        elif state=='memory_event':
            prompt+='本次没有新口头问题：你之前问过他一件事一直没得到回答，现在他在场且安静。可以轻轻提一次，语气自然，不要责怪也不要重复追问。'
        elif speaker.get('held_utterance') or (trigger.get('kind')=='held'):
            prompt+=('这句话是他刚才说的，当时不确定是不是在对你说，所以没马上回；现在他的注意力到你身上了，'
                     '就当他刚跟你说完这句来回答，不要提"刚才没听清""你是不是在跟我说话"。')
        elif trigger.get('room_recently_active'):
            # Turning to the robot out of a conversation is a question about the
            # conversation. Greeting somebody mid-discussion is the wrong move
            # even if they personally have not spoken to you yet.
            prompt+=('本次没有新口头问题：刚才房间里有人在说话，现在有人转向你。接住 recent 里刚刚在聊的那件事说一句，'
                     '不要打招呼、不要问"想聊点什么"、不要说谁刚回来。')
        elif trigger.get('spoke_this_session') is False:
            prompt+='本次没有新口头问题：这个人正看着你，本次会话还没说过话。用一句自然的招呼开场，可以结合history里你们以前聊过的话题，但不要假装他刚说了什么，也不要复述旧对话。'
        else:
            prompt+=('本次没有新口头问题：有人正看着你，像在等你开口。'
                     '依据recent里刚才的对话接一句有内容的话——对他们讨论的一点看法、一句具体的关心或提议都可以；'
                     '不要问"需要我帮忙吗""你想聊什么"，不要复述他们的原话，也不要自我介绍。'
                     'recent为空时，用一句自然的招呼开场。')
    if speaker.get('dialogue_link'):
        link=speaker['dialogue_link']
        context['answering_robot_question']={'question':link['question'],'person_id':link['person_id'],'previous_turn_id':link['turn_id'],
            'preceding_conversation':(link.get('context') or {}).get('recent_conversation',[])}
        prompt+='当前发言是同一用户对你刚才问题的回答，即使视线转开也延续这轮；结合你问的问题与先前话题理解确认、否定或补充，不重新问候。'
    if sensor:context['perception']=sensor
    if proactive:
        prompt+=('现在是你主动开口，对象就是 owner_observation 里的那个人，用他的名字或"你"称呼，不要叫"主人"。'
                 '用确有依据的历史话题说一句，最多两句，不把旧短应答当新请求。')
        context['owner_observation']={k:proactive[k] for k in ['resolved_person_id','role','emotion_valid','emotion_label'] if k in proactive}
    # The session id is already implied (anything from another one is flagged),
    # and a raw millisecond stamp is six tokens the model cannot read. Wall clock
    # time it can: greetings, "下午三点", "今天" all depend on it.
    if session_block.get('session_id'):
        detail=session_block.get('participant_detail') or {}
        def who(key):
            return (detail.get(key) or {}).get('label') or key
        roster=[]
        for key in (session_block.get('participants') or []):
            row=detail.get(key) or {}
            roster.append(compact_turn({'name':who(key),
                'identity_confirmed':True if row.get('identity') else False,
                'observed_now':key in (session_block.get('observed_participants') or []) or None,
                'on_camera':key in (session_block.get('visible_participants') or []) or None}))
        context['session']=compact_turn({'session_id':session_block.get('session_id'),
            'participants':[item['name'] for item in roster],
            'who':roster or None,
            'not_observed_now':[who(p) for p in (session_block.get('participants') or [])
                                if p not in (session_block.get('observed_participants') or [])] or None,
            'robot_involved':session_block.get('robot_involved'),'state':session_block.get('state'),
            # The window, in seconds the model can read: when this conversation
            # started and how long since the last line. Silence inside it is
            # still the same conversation — that is what DORMANT means.
            'started_t':ago(session_block.get('start_ms')),
            'last_line_t':ago(session_block.get('last_activity_ms')),
            'called_by_name_t':ago(session_block.get('wake_word_ms')) if session_block.get('wake_word_ms') else None})
    context["current_turn"]=compact_turn({"utterance_id":speaker.get("utterance_id"),
        "time":time.strftime("%H:%M",time.localtime(now/1000.)) if now else None})
    # When `recent` is a session transcript it is not a 120-second slice of the
    # room; it is this conversation, in order. Saying "window 120000" then would
    # invite the model to treat anything older as missing and re-ask.
    from_session=bool(session_block.get('session_id')) and bool(recent_sorted) and all(r.get('order') for r in recent_out)
    context["context_limits"]={"recent_max":16,"history_max":12,"history_is_retrieved_subset":True,
                               "t_unit":"秒，相对当前发言，负数为更早","visual_observations_only_on_latest_turns":True}
    if from_session:context["context_limits"]["recent_is"]="本场对话的完整记录，按 order 排序"
    else:context["context_limits"]["recent_window_ms"]=120000
    context["current_user_text"]=user_text
    return fit_to_budget(prompt,context)
