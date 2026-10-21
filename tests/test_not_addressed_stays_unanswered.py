"""注意力判定不是对机器人说的，就不该被回答。

Straight from the field trace of 2026-09-20 19:30. 天行 said 「我感觉肚子有点
饿。」 — the addressee gate scored it 0, no evidence at all, decision `ignore`,
「不是对机器人说的」 — and five seconds later the robot said 「饿了就找点吃的，
别硬扛。」 It reached the model through the other door: attention settled on him,
and the rule 「接他刚才说的那句」 took any unanswered sentence from the last two
minutes, whether or not it had ever been for the robot.

Two things have to be true at once, and they pull in opposite directions:

    * somebody who talks to himself and *then turns to the robot* should have
      that sentence picked up — asking him to repeat it is the behaviour that
      made the robot feel deaf;
    * a sentence aimed at another person is never the robot's to answer, and
      neither is one from a minute ago that nobody has invited it into.
"""
import time
import unittest
from unittest.mock import Mock, patch

from brain.runtime import BrainRuntime


def heard(text, stamp, person='天行', granted=False, held=False, terms=None, addressee=None):
    return {'utterance_id': 'h' + str(stamp), 'person_id': person, 'participant_id': person,
            'stamp_ms': stamp, 'held_for_attention': held,
            'addressee': addressee or ('robot' if granted else 'non_robot_or_unknown'),
            'transcript': {'text': text, 'is_final': True, 'emitted_ms': stamp, 'ended_ms': stamp},
            'attention': {'addressed_to_robot': granted, 'held_for_attention': held,
                          'reasons': [], 'addressee_detail': {'terms': terms or {}}}}


def engagement(stamp, person='天行', state='INVITED', reasons=('gaze_engaged', 'silent_invitation')):
    return {'attention': {'target_id': 'person:' + person, 'person_id': person, 'listen': True,
                          'confidence': .72, 'interaction_phase': 'LISTENING'},
            'workspace_events': [{'kind': 'engagement_event', 'person_id': person, 'share': .6,
                                  'engagement_state': state, 'reasons': list(reasons),
                                  'summary': None, 'detail': None}]}


class AmbientTalkTests(unittest.TestCase):
    def runtime(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '好的。'
        runtime.memory.search = Mock(return_value=[])
        self.addCleanup(runtime.stop)
        return runtime

    def settle(self, runtime):
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.01)

    def test_a_sentence_aimed_at_another_person_is_never_picked_up(self):
        runtime = self.runtime()
        runtime.state['heard'] = [heard('老王你等会儿过来一下', 100000,
                                        terms={'addresses_another_person': -.8})]
        self.assertEqual(runtime._recent_unanswered('天行', 101000, engaged=True), '')

    def test_nor_is_one_said_while_somebody_else_held_the_floor(self):
        runtime = self.runtime()
        turn = heard('那就这样吧', 100000)
        turn['attention']['reasons'] = ['another_speaker_has_the_floor']
        runtime.state['heard'] = [turn]
        self.assertEqual(runtime._recent_unanswered('天行', 101000, engaged=True), '')

    def test_talking_about_the_robot_to_somebody_else_is_not_talking_to_it(self):
        runtime = self.runtime()
        runtime.state['heard'] = [heard('它现在特别敏感', 100000,
                                        terms={'third_person_about_robot': -.5})]
        self.assertEqual(runtime._recent_unanswered('天行', 101000, engaged=True), '')

    def test_self_talk_is_picked_up_when_he_turns_to_the_robot(self):
        """这一条是 9 月 18 号明确要的：自言自语后回头，它要接住那句话。"""
        runtime = self.runtime()
        runtime.state['heard'] = [heard('我感觉肚子有点饿', 100000)]
        self.assertEqual(runtime._recent_unanswered('天行', 104000, engaged=True), '我感觉肚子有点饿')

    def test_but_not_when_nobody_is_engaging_it(self):
        """房间里有人、机器人自己想开口，不等于可以去翻别人刚才说的话。"""
        runtime = self.runtime()
        runtime.state['heard'] = [heard('我感觉肚子有点饿', 100000)]
        self.assertEqual(runtime._recent_unanswered('天行', 104000, engaged=False), '')

    def test_and_not_a_sentence_from_a_minute_ago(self):
        runtime = self.runtime()
        runtime.state['heard'] = [heard('我感觉肚子有点饿', 100000)]
        self.assertEqual(runtime._recent_unanswered('天行', 100000 + 90000, engaged=True), '')

    def test_a_sentence_the_gate_kept_for_us_survives_the_full_window(self):
        """门控记下来的那种（分数够高、只是当时不确定）不受 30 秒限制。"""
        runtime = self.runtime()
        runtime.state['heard'] = [heard('周末去哪走走', 100000, held=True)]
        self.assertEqual(runtime._recent_unanswered('天行', 100000 + 90000, engaged=True),
                         '周末去哪走走')

    def test_quiet_company_does_not_answer_what_was_never_ours(self):
        """整条链路：门控判了 ignore，机器人只是注意到他，不许回。"""
        runtime = self.runtime()
        stamp = int(time.time() * 1000)
        runtime.state['heard'] = [heard('我感觉肚子有点饿', stamp - 5000)]
        runtime.step({'attention': {'target_id': 'person:天行', 'person_id': '天行', 'listen': False},
                      'workspace_events': [{'kind': 'presence_event', 'person_id': '天行',
                                            'share': .4, 'engagement_state': 'presence_event',
                                            'summary': '他在这儿待了一会儿', 'reasons': ['quiet_company']}]})
        self.settle(runtime)
        # 那句话仍然在上下文里（它是背景，本来就该看得见），但它不是这一轮
        # 要回答的问题：user_text 是空的，开口的理由也不是「接他刚说的那句」。
        self.assertEqual(runtime.llm.respond.call_args.args[0], '')
        self.assertNotEqual(runtime.state['current_turn']['speak_because'], 'their_recent_words')


if __name__ == '__main__':
    unittest.main()
