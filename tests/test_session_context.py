"""语境送到模型那一刻长什么样。

The session layer lives in robot-attention-perception and is tested there. What
these tests are about is the last hop: the conversation Brain hands the model.
Three things have to be true of it, and each of them was false at some point in
the field —

    * `recent` is this conversation, not a 120-second slice of the room, so a
      line from the lunch group two metres away never arrives as context;
    * the roster is everybody in the session, including whoever is silent or
      off camera, so 「老王你怎么看」 is not answered as if 老王 were not there;
    * the order is explicit and includes the robot's own turns, because
      without them it answered 「上海迪士尼挺好玩的」 with the same sentence back.
"""
import json
import time
import unittest
from unittest.mock import Mock

from brain.prompt_context import build_context
from brain.runtime import BrainRuntime


def a_turn(stamp, text='那你觉得呢'):
    return {'utterance_id': 'now', 'person_id': '天行', 'stamp_ms': stamp,
            'transcript': {'text': text, 'started_ms': stamp - 800, 'ended_ms': stamp,
                           'is_final': True, 'emitted_ms': stamp,
                           'speaker': {'speaker_id': '天行', 'similarity': .8}},
            'attention': {'listen': True, 'addressed_to_robot': True, 'confidence': .9,
                          'reasons': ['addressed_form']}}


def heard(utterance_id, person, text, stamp):
    return {'utterance_id': utterance_id, 'person_id': person, 'stamp_ms': stamp,
            'transcript': {'text': text, 'started_ms': stamp - 500, 'ended_ms': stamp},
            'attention': {'addressed_to_robot': False}}


class SessionContextReachesTheModelTests(unittest.TestCase):
    def runtime(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '那就周六去。'
        runtime.memory.search = Mock(return_value=[])
        self.addCleanup(runtime.stop)
        return runtime

    def settle(self, runtime):
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.01)
        return runtime.state['current_turn']['model_context']

    def test_the_session_transcript_replaces_the_room_window(self):
        runtime = self.runtime()
        stamp = int(time.time() * 1000)
        # The room heard both conversations; only one of them is ours.
        runtime.state['heard'] = [heard('lunch_2', '陌生人2', '楼下那家吧', stamp - 4000),
                                  heard('lunch_1', '陌生人1', '中午吃什么', stamp - 6000),
                                  heard('ours_1', '天行', '周末想去哪走走', stamp - 20000)]
        session = {
            'session_id': 'session_003', 'state': 'ACTIVE', 'robot_involved': True,
            'participants': ['天行', '陌生人3'], 'observed_participants': ['天行'],
            'visible_participants': ['天行'],
            'transcript': [{'i': 1, 'utterance_id': 'ours_1', 'speaker': '天行',
                            'text': '周末想去哪走走', 'stamp_ms': stamp - 20000, 'to_robot': False},
                           {'i': 2, 'utterance_id': 'ours_2', 'speaker': '陌生人3',
                            'text': '公园吧', 'stamp_ms': stamp - 15000, 'to_robot': False}]}

        runtime.step({'utterances': [a_turn(stamp)], 'attention': {'target_id': '天行', 'listen': True},
                      'session_context': session})
        context = self.settle(runtime)

        lines = [row['text'] for row in context['recent']]
        self.assertEqual(lines, ['周末想去哪走走', '公园吧'])
        self.assertNotIn('中午吃什么', lines)
        self.assertNotIn('楼下那家吧', lines)
        self.assertEqual([row['order'] for row in context['recent']], [1, 2])
        # …and the model is told this is the conversation, not a time slice.
        self.assertNotIn('recent_window_ms', context['context_limits'])

    def test_the_roster_keeps_the_silent_and_the_off_camera_member(self):
        runtime = self.runtime()
        stamp = int(time.time() * 1000)
        session = {
            'session_id': 'session_004', 'state': 'ACTIVE', 'robot_involved': True,
            'participants': ['天行', '老王', '陌生人5'],
            'observed_participants': ['天行', '老王'],     # 老王 is heard, not seen
            'visible_participants': ['天行'],
            'transcript': [{'i': 1, 'speaker': '天行', 'text': '老王你怎么看',
                            'stamp_ms': stamp - 9000, 'to_robot': False},
                           {'i': 2, 'speaker': '老王', 'text': '我觉得没问题',
                            'stamp_ms': stamp - 6000, 'to_robot': False}]}

        runtime.step({'utterances': [a_turn(stamp, '小圆你怎么看')],
                      'attention': {'target_id': '天行', 'listen': True},
                      'session_context': session})
        context = self.settle(runtime)

        self.assertEqual(context['session']['participants'], ['天行', '老王', '陌生人5'])
        # 陌生人5 has said nothing and is not observed this instant; still a member.
        self.assertEqual(context['session']['not_observed_now'], ['陌生人5'])
        self.assertTrue(context['session']['robot_involved'])
        self.assertEqual(context['session']['state'], 'ACTIVE')

    def test_the_robots_own_replies_are_in_the_transcript_in_order(self):
        runtime = self.runtime()
        stamp = int(time.time() * 1000)
        runtime.state['events'] = [{'text': '我也觉得公园不错。', 'stamp_ms': stamp - 12000}]
        session = {
            'session_id': 'session_005', 'state': 'ACTIVE', 'robot_involved': True,
            'participants': ['天行'], 'observed_participants': ['天行'],
            'transcript': [{'i': 1, 'speaker': '天行', 'text': '周末去公园吧',
                            'stamp_ms': stamp - 16000, 'to_robot': True},
                           {'i': 2, 'speaker': '天行', 'text': '你还记得上次那条路吗',
                            'stamp_ms': stamp - 8000, 'to_robot': True}]}

        runtime.step({'utterances': [a_turn(stamp)], 'attention': {'target_id': '天行', 'listen': True},
                      'session_context': session})
        context = self.settle(runtime)

        said = [(row['order'], row['speaker'], row['text']) for row in context['recent']]
        self.assertEqual(said, [(1, '天行', '周末去公园吧'),
                                (2, '小圆', '我也觉得公园不错。'),
                                (3, '天行', '你还记得上次那条路吗')])

    def test_without_a_session_the_room_window_still_works(self):
        """会话层没数据时不能把上下文弄丢：回到原来的 120 秒窗口。"""
        runtime = self.runtime()
        stamp = int(time.time() * 1000)
        runtime.state['heard'] = [heard('prior', '天行', '周末想去哪走走', stamp - 9000)]
        session = {}

        runtime.step({'utterances': [a_turn(stamp)], 'attention': {'target_id': '天行', 'listen': True},
                      'session_context': session})
        context = self.settle(runtime)

        self.assertIn('周末想去哪走走', [row['text'] for row in context['recent']])
        self.assertEqual(context['context_limits']['recent_window_ms'], 120000)
        self.assertNotIn('session', context)


class IdentityIsSpelledOutTests(unittest.TestCase):
    """谁说的、确认没确认，提示词里必须写明白。

    The field case: 天行's short sentences were not recognised, went into the
    prompt as speaker `unknown`, and the model decided from context that
    `unknown` was 天行. It happened to be right. With a second person in the
    room the same reasoning hands somebody else's answer to the wrong person,
    so the system says what it knows and how well it knows it.
    """

    def context(self, rows, session=None):
        speaker = {'person_id': '天行', 'stamp_ms': 1789551300000, 'session_id': 'scope',
                   'session': session or {'session_id': 'session_009',
                                          'participants': ['天行', 'participant_002'],
                                          'observed_participants': ['天行'],
                                          'visible_participants': ['天行'],
                                          'participant_detail': {
                                              '天行': {'label': '天行', 'identity': '天行',
                                                       'identity_confidence': .62},
                                              'participant_002': {'label': '说话人2', 'identity': '',
                                                                  'identity_confidence': 0.0}},
                                          'state': 'ACTIVE', 'robot_involved': True,
                                          'start_ms': 1789551200000,
                                          'last_activity_ms': 1789551295000},
                   'recent_conversation': rows}
        _, context = build_context('那你觉得呢', [], None, {'current_speaker': speaker})
        return context

    def test_an_inferred_line_is_marked_as_inferred(self):
        context = self.context([
            {'utterance_id': 'a', 'person_id': '天行', 'text': '去了。', 'order': 1,
             'stamp_ms': 1789551290000, 'identity_status': 'confirmed'},
            {'utterance_id': 'b', 'person_id': '天行', 'text': '一般。', 'order': 2,
             'stamp_ms': 1789551295000, 'identity_status': 'inferred_session_voice'}])
        first, second = context['recent']
        self.assertNotIn('identity_note', first)
        self.assertIn('本场声纹比对', second['identity_note'])

    def test_an_unconfirmed_speaker_is_a_numbered_person_not_unknown(self):
        context = self.context([
            {'utterance_id': 'b', 'person_id': '说话人2', 'text': '他说什么？', 'order': 1,
             'stamp_ms': 1789551295000, 'identity_status': 'unconfirmed'}])
        row = context['recent'][0]
        self.assertEqual(row['speaker'], '说话人2')
        self.assertIn('身份未确认', row['identity_note'])
        self.assertNotIn('unknown', json.dumps(context, ensure_ascii=False))

    def test_the_roster_says_who_is_confirmed_and_who_is_not(self):
        context = self.context([
            {'utterance_id': 'a', 'person_id': '天行', 'text': '去了。', 'order': 1,
             'stamp_ms': 1789551290000, 'identity_status': 'confirmed'}])
        block = context['session']
        self.assertEqual(block['participants'], ['天行', '说话人2'])
        self.assertEqual([row['identity_confirmed'] for row in block['who']], [True, False])
        # 说话人2 is a member the sensors cannot see or hear right now.
        self.assertEqual(block['not_observed_now'], ['说话人2'])

    def test_the_window_reaches_the_model_as_seconds(self):
        context = self.context([
            {'utterance_id': 'a', 'person_id': '天行', 'text': '去了。', 'order': 1,
             'stamp_ms': 1789551290000, 'identity_status': 'confirmed'}])
        block = context['session']
        self.assertEqual(block['started_t'], -100)     # 这场 100 秒前开始
        self.assertEqual(block['last_line_t'], -5)     # 最后一句 5 秒前
        self.assertNotIn('start_ms', json.dumps(block))


class EarlierSessionFlagTests(unittest.TestCase):
    """本场的话不能被标成「上一场」。

    The flag exists so the model does not answer a question from an hour ago.
    There are two session ids in play — the process scope and the conversation —
    and comparing against only one of them marked the whole current transcript
    `earlier_session: True`, which is an instruction to ignore it.
    """

    def context(self, rows, session_id='session_007', scope='mac-first-test'):
        speaker = {'person_id': '天行', 'stamp_ms': 1789551300000, 'session_id': scope,
                   'session': {'session_id': session_id, 'participants': ['天行'],
                               'observed_participants': ['天行'], 'state': 'ACTIVE'},
                   'recent_conversation': rows}
        _, context = build_context('那你觉得呢', [], None, {'current_speaker': speaker})
        return context

    def test_lines_of_this_conversation_are_not_flagged(self):
        context = self.context([{'utterance_id': 'a', 'person_id': '天行', 'text': '周末去公园吧',
                                 'stamp_ms': 1789551290000, 'order': 1, 'session_id': 'session_007'}])
        self.assertNotIn('earlier_session', context['recent'][0])

    def test_nor_are_lines_carrying_the_process_scope(self):
        context = self.context([{'utterance_id': 'a', 'person_id': '天行', 'text': '周末去公园吧',
                                 'stamp_ms': 1789551290000, 'session_id': 'mac-first-test'}])
        self.assertNotIn('earlier_session', context['recent'][0])

    def test_but_a_genuinely_older_session_still_is(self):
        context = self.context([{'utterance_id': 'a', 'person_id': '天行', 'text': '上次说的那个方案',
                                 'stamp_ms': 1789551200000, 'session_id': 'session_001'}])
        self.assertTrue(context['recent'][0]['earlier_session'])


if __name__ == '__main__':
    unittest.main()
