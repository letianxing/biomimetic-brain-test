#!/usr/bin/env python3
"""Replay the hard conversational cases against the real model, and print what it says.

This is the answer to "how do I tell whether it replies correctly": the cases the
robot kept getting wrong in the field are written down here as fixed perception
input, sent through the same build_context and the same LLM client the robot
uses, and printed with the checks that made them wrong. No microphone, no camera,
no speaker — only the remote model is real.

    BRAIN_LLM_BASE_URL=http://172.16.60.162:18050/v1 BRAIN_LOCATION=深圳 \
      ~/Golands/robot-attention-perception/.venv-mac/bin/python scripts/verify_reply_quality.py

What it does not do: it does not score the robot. Each case prints PASS/FAIL on a
blunt textual check (did it open with a question, did it use the right speaker's
words, did the weather answer contain a number). Those catch the failures we
actually saw. They are not a quality metric and a PASS is not accuracy.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.llm import configured_llm

NOW = int(time.time() * 1000)


def turn(person, text, seconds_ago, to_robot=False):
    return {'person_id': person, 'text': text, 'stamp_ms': NOW - seconds_ago * 1000,
            'to_robot': to_robot, 'session_id': 'quality-check',
            'background': {'visible_participants': 3, 'addressee': 'group' if not to_robot else 'robot'}}


BYSTANDERS = [
    turn('天行', '我觉得下午三点那场评审应该提前，不然场地要还给他们了', 42),
    turn('二丫', '提前也行，但是数据还没核对完，提前了反而讲不清楚', 34),
    turn('天行', '数据我昨天晚上核过一遍了，主要是第三张表的口径不一样', 26),
    turn('二丫', '那口径这事你得先跟他们对一下，不然评审现场又要吵', 18),
]


def speaker(person, extra=None):
    payload = {'person_id': person, 'session_id': 'quality-check', 'stamp_ms': NOW,
               'recent_conversation': BYSTANDERS, 'attention': {}, 'voice': {'speaker_role': 'owner'},
               'group_context': {'active': True, 'participants': ['天行', '二丫', 'robot']}}
    payload.update(extra or {})
    return {'current_speaker': payload}


def no_leading_question(text):
    head = text.strip()[:14]
    return not any(marker in head for marker in ('你是想问', '你指的是', '你是说', '是想问', '你想问'))


def mentions(text, words):
    return any(word in text for word in words)


CASES = [
    {'name': '旁人聊完，一个人转头问「我刚说的对吗」',
     'user_text': '你说我刚刚说的对吗',
     'perception': speaker('二丫'),
     'checks': [('不以反问开场', no_leading_question),
                ('回到二丫说的口径/对一下这件事', lambda t: mentions(t, ('口径', '对一下', '核对', '评审', '数据')))]},
    {'name': '同样的旁人对话，另一个人问「你觉得呢」',
     'user_text': '你觉得呢',
     'perception': speaker('天行'),
     'checks': [('不以反问开场', no_leading_question),
                ('给出立场而不是复述', lambda t: mentions(t, ('提前', '口径', '先', '建议', '可以', '不如')))]},
    {'name': '没有人说话，有人一直看着机器人（无声邀请）',
     'user_text': '',
     'perception': speaker('天行', {'attention_trigger': {'kind': 'silent_invitation', 'engagement_state': 'INVITED',
                                                          'person_id': '天行', 'spoke_this_session': True}}),
     'checks': [('不说「需要我帮忙吗」这类空话', lambda t: not mentions(t, ('需要我帮忙', '你想聊什么', '有什么可以帮'))),
                ('接住刚才的话题', lambda t: mentions(t, ('评审', '口径', '数据', '三点', '场地')))]},
    {'name': '问天气（要调用工具，不能反问城市）',
     'user_text': '今天天气怎么样',
     'perception': speaker('天行', {'recent_conversation': []}),
     'checks': [('不反问是哪个城市', lambda t: not mentions(t, ('哪个城市', '你在哪', '想问哪'))),
                ('答案里有温度或天气词', lambda t: any(ch.isdigit() for ch in t) or mentions(t, ('晴', '云', '雨', '雪', '阴')))]},
    {'name': '直接问事实：1+1（不许反问）',
     'user_text': '1+1等于几',
     'perception': speaker('天行', {'recent_conversation': []}),
     'checks': [('直接给出答案 2', lambda t: '2' in t or '二' in t),
                ('不反问', no_leading_question)]},
    {'name': '主人主动搭话：不能把「主人」和本人说成两个人',
     'user_text': '',
     'perception': speaker('天行', {'recent_conversation': [turn('天行', '明天是周六了', 30, to_robot=True)],
                                    'attention_trigger': {'kind': 'presence_event', 'engagement_state': 'presence_event',
                                                          'person_id': '天行', 'spoke_this_session': True,
                                                          'summary': '他在这儿有一会儿了，房间一直很安静',
                                                          'detail': {'present_ms': 90000, 'silent_ms': 60000,
                                                                     'expression': 'neutral'}}}),
     'checks': [('不用「主人」当称呼', lambda t: '主人' not in t),
                ('不把本人说成第三人称「天行说」', lambda t: '天行说' not in t and '天行刚才' not in t),
                ('不问「需要我帮忙吗」', lambda t: not mentions(t, ('需要我帮忙', '你想聊什么', '有什么可以帮')))]},
    {'name': '「你猜我是谁呀」——不许把问题弹回去',
     'user_text': '小圆，你猜我是谁呀',
     'perception': speaker('天行', {'recent_conversation': []}),
     'checks': [('答的是对方是谁', lambda t: '天行' in t),
                ('不是自我介绍', lambda t: not t.strip().startswith(('我是小圆', '我叫小圆'))),
                ('不把问句原样弹回', lambda t: '你猜我是谁' not in t)]},
    {'name': '「我感觉很烦」——要接住情绪，不是反问',
     'user_text': '我说我感觉很烦',
     'perception': speaker('天行', {'recent_conversation': [turn('小圆', '天行，我在这儿，慢慢说。', 20, to_robot=False)]}),
     'checks': [('不以反问开场', no_leading_question),
                # 现场的毛病是「一模一样的一句话说第二遍」，不是不许安慰
                ('不是上一句的原样复读', lambda t: t.strip() != '天行，我在这儿，慢慢说。'),
                ('接住了情绪这件事', lambda t: mentions(t, ('烦', '怎么了', '发生', '心情', '不开心')))]},
    # Deliberately absent: "二丫你觉得这样行吗". Whether that reaches the model at
    # all is the attention gate's decision, not the model's — by the time text is
    # here it has already been judged to be addressed to the robot, and the model
    # is right to answer it. That case belongs to
    # robot-attention-perception/scripts/verify_attention_algorithm.py, which
    # checks it stays IDLE.
]


def main():
    client = configured_llm()
    if client is None:
        print('BRAIN_LLM_BASE_URL 没设置，无法跑真实模型。')
        return 1
    if not os.environ.get('BRAIN_LOCATION'):
        print('提示：BRAIN_LOCATION 没设置，天气那条只会回答「不知道在哪」，这是设计如此。\n')
    failures = 0
    for case in CASES:
        started = time.perf_counter()
        try:
            reply = client.respond(case['user_text'], [], perception=case['perception'])
        except Exception as exc:
            print(f"[{case['name']}] 调用失败: {exc}")
            failures += 1
            continue
        elapsed = (time.perf_counter() - started) * 1000
        tools = [item['name'] for item in client.last_tool_results]
        print(f"■ {case['name']}")
        print(f"  用户说: {case['user_text'] or '（没说话，只是看着）'}")
        print(f"  小圆答: {reply}")
        print(f"  {elapsed:.0f}ms" + (f"  工具: {tools}" if tools else ''))
        for label, check in case['checks']:
            ok = check(reply)
            failures += 0 if ok else 1
            print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
        print()
    print('这些检查只说明这几句文本符合/不符合我们写下的要求，不是识别率，也不是对话质量评分。')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
