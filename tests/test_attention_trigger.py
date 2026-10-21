import time
import unittest
from unittest.mock import Mock, patch

from brain.attention_trigger import AttentionTrigger
from brain.runtime import BrainRuntime


def attention(stamp, person='a', state='INVITED', transition=7, **extra):
    return {'stamp_ms': stamp, 'person_id': person, 'target_kind': 'person',
            'interaction_phase': 'INVITED', 'listen': True, 'confidence': .9,
            'transition_id': transition,
            'engagement': {'state': state, 'person_id': person, 'confidence': .79,
                           'reasons': ['gaze_engaged', 'memory_participant', 'silent_invitation']},
            **extra}


class TriggerTests(unittest.TestCase):
    def test_fires_once_then_needs_somebody_to_actually_say_something(self):
        trigger = AttentionTrigger()
        first = trigger.update({'attention': attention(3000)}, 3000, participants={'a': 1}, last_heard_ms=2000)
        self.assertEqual(first['kind'], 'attention_engagement')
        self.assertEqual(first['engagement_state'], 'INVITED')
        self.assertIsNone(trigger.update({'attention': attention(3200)}, 3200, participants={'a': 1}, last_heard_ms=2000))
        # Long after the interval, but still nothing new was said.
        self.assertIsNone(trigger.update({'attention': attention(30000)}, 30000, participants={'a': 1}, last_heard_ms=2000))
        self.assertEqual(trigger.status()['reason'], 'nothing_new_since_our_last_turn')
        # Somebody speaks; now the context is new and it may invite again.
        self.assertIsNotNone(trigger.update({'attention': attention(31000)}, 31000, participants={'a': 1}, last_heard_ms=30500))

    def test_the_robot_speaking_cannot_re_arm_it(self):
        """The regression this guards: attention leaves and re-enters the invited
        phase every time the robot finishes a reply, which used to look like a
        brand new episode and produced the same sentence on a loop. Neither a new
        transition id nor the robot's own voice may re-arm it."""
        trigger = AttentionTrigger()
        self.assertIsNotNone(trigger.update({'attention': attention(3000, transition=7)}, 3000, participants={'a': 1}, last_heard_ms=2000))
        for stamp, transition in ((13000, 8), (23000, 9), (33000, 10)):
            state = {'attention': attention(stamp, transition=transition)}
            self.assertIsNone(trigger.update(state, stamp, participants={'a': 1}, last_heard_ms=2000),
                              f'a new transition id at {stamp} must not re-arm the trigger')

    def test_a_backstop_interval_holds_even_while_people_keep_talking(self):
        trigger = AttentionTrigger()
        self.assertIsNotNone(trigger.update({'attention': attention(3000)}, 3000, participants={'a': 1}, last_heard_ms=2000))
        self.assertIsNone(trigger.update({'attention': attention(5000)}, 5000, participants={'a': 1}, last_heard_ms=4000))
        self.assertEqual(trigger.status()['reason'], 'min_interval')
        self.assertIsNotNone(trigger.update({'attention': attention(12000)}, 12000, participants={'a': 1}, last_heard_ms=11000))

    def test_opening_on_somebody_who_has_not_spoken_needs_a_quiet_room(self):
        """Moved out of Brain's owner rule: starting a conversation with somebody
        who has said nothing yet must wait for the room to settle, whoever they
        are. Joining one that is already running does not."""
        trigger = AttentionTrigger()
        self.assertIsNone(trigger.update({'attention': attention(3000)}, 3000, last_heard_ms=2000))
        self.assertEqual(trigger.status()['reason'], 'arrival_needs_a_quiet_room')
        self.assertIsNotNone(trigger.update({'attention': attention(9000)}, 9000, last_heard_ms=2000))

    def test_busy_stale_and_other_states_do_not_fire(self):
        self.assertIsNone(AttentionTrigger().update({'attention': attention(3000)}, 3000, busy=True))
        self.assertIsNone(AttentionTrigger().update({'attention': attention(1000)}, 3000))
        self.assertIsNone(AttentionTrigger().update({'attention': attention(3000, state='OBSERVING')}, 3000))
        self.assertIsNone(AttentionTrigger().update({'attention': attention(3000, state='')}, 3000))

    def test_participants_come_from_brain_not_invented(self):
        result = AttentionTrigger().update({'attention': attention(3000)}, 3000, participants={'a': 1, 'b': 2})
        self.assertEqual(result['participants'], ['a', 'b'])
        self.assertEqual(AttentionTrigger().update({'attention': attention(3000)}, 3000)['participants'], [])


class RuntimeTests(unittest.TestCase):
    """The silent invitation now arrives as an attention state, not a Brain rule."""

    def test_engagement_state_sends_the_real_discussion_to_the_model(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '要一起定个时间吗？'
        runtime.memory.search = Mock(return_value=[])
        self.addCleanup(runtime.stop)

        def heard(key, person, text, stamp):
            return {'utterance_id': key, 'person_id': person, 'stamp_ms': stamp, 'temporal_group_id': 'group',
                    'transcript': {'text': text, 'is_final': True, 'emitted_ms': stamp},
                    'attention': {'listen': False, 'addressed_to_robot': False}}

        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step({'utterances': [heard('a', 'a', '周六去公园', 98000), heard('b', 'b', '还没定时间', 99000)]})
        runtime.tts.speak.assert_not_called()

        with patch('brain.runtime.now_ms', return_value=101000):
            runtime.step({'attention': attention(101000)})

        deadline = time.monotonic() + 2
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(runtime.jobs)
        runtime.llm.respond.assert_called_once()

        call = runtime.llm.respond.call_args
        speaker = call.kwargs['perception']['current_speaker']
        # What he said a moment ago is what the reply is about. Opening on
        # nothing and letting the model pick something out of the context is
        # how the field got greeted as a new arrival by a robot that had just
        # heard him say 我今天身体不太舒服。想出去玩。
        self.assertEqual(call.args[0], '周六去公园')
        self.assertEqual(runtime.state['current_turn']['speak_because'], 'their_recent_words')
        self.assertEqual(speaker['attention_trigger']['kind'], 'attention_engagement')
        self.assertEqual(speaker['attention_trigger']['engagement_state'], 'INVITED')
        self.assertEqual([turn['text'] for turn in speaker['recent_conversation']], ['周六去公园', '还没定时间'])
        self.assertTrue(runtime.state['current_turn']['model_context'].get('attention_trigger'))

    def test_a_reply_already_in_progress_blocks_the_trigger(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.llm = Mock()
        self.addCleanup(runtime.stop)
        runtime.state['s2s']['speaking'] = True
        with patch('brain.runtime.now_ms', return_value=101000):
            runtime.step({'attention': attention(101000)})
        self.assertFalse(runtime.jobs)
        runtime.llm.respond.assert_not_called()


if __name__ == '__main__':
    unittest.main()


class WorkspaceEventTests(unittest.TestCase):
    """Attention can also raise something nobody is doing right now."""

    def state(self, stamp, kind='affect_event', person='a'):
        return {'attention': {'stamp_ms': stamp, 'engagement': {'state': 'OBSERVING', 'person_id': person}},
                'workspace_events': [{'candidate_id': f'{kind}:1', 'kind': kind, 'person_id': person,
                                      'share': .42, 'admitted': True, 'summary': '表情持续偏低，且已安静一段时间',
                                      'detail': {'samples': 18, 'negative_share': .78},
                                      'reasons': ['sustained_low_expression', 'room_quiet']}]}

    def test_an_affect_candidate_can_open_a_turn_and_carries_its_evidence(self):
        trigger = AttentionTrigger()
        result = trigger.update(self.state(3000), 3000, last_heard_ms=2000)
        self.assertIsNotNone(result)
        self.assertEqual(result['engagement_state'], 'affect_event')
        self.assertEqual(result['person_id'], 'a')
        self.assertEqual(result['detail']['samples'], 18)
        self.assertIn('sustained_low_expression', result['reasons'])

    def test_a_memory_candidate_works_the_same_way(self):
        result = AttentionTrigger().update(self.state(3000, kind='memory_event'), 3000, last_heard_ms=2000)
        self.assertEqual(result['engagement_state'], 'memory_event')

    def test_the_same_guards_apply_to_workspace_candidates(self):
        self.assertIsNone(AttentionTrigger().update(self.state(3000), 3000, busy=True, last_heard_ms=2000))
        trigger = AttentionTrigger()
        self.assertIsNotNone(trigger.update(self.state(3000), 3000, last_heard_ms=2000))
        self.assertIsNone(trigger.update(self.state(30000), 30000, last_heard_ms=2000),
                          'nothing new was said, so it must not raise it again')

    def test_an_unknown_event_kind_is_ignored(self):
        state = self.state(3000)
        state['workspace_events'][0]['kind'] = 'sensory'
        self.assertIsNone(AttentionTrigger().update(state, 3000, last_heard_ms=2000))


class AffectRuntimeTests(unittest.TestCase):
    """Owner present and silent, expression persistently low: end to end."""

    def payload(self, stamp):
        return {'attention': {'stamp_ms': stamp, 'engagement': {'state': 'OBSERVING', 'person_id': 'owner'}},
                'workspace_events': [{'candidate_id': 'affect:owner:1', 'kind': 'affect_event',
                                      'person_id': 'owner', 'share': .38, 'admitted': True,
                                      'summary': '表情持续偏低，且已安静一段时间',
                                      'detail': {'samples': 18, 'negative_share': .78,
                                                 'mean_valence': -.42, 'silent_ms': 31000},
                                      'reasons': ['sustained_low_expression', 'room_quiet']}]}

    def test_it_opens_a_turn_and_the_prompt_forbids_asserting_a_mood(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '在想什么呢？'
        runtime.memory.search = Mock(return_value=[])
        self.addCleanup(runtime.stop)

        with patch('brain.runtime.now_ms', return_value=200000):
            runtime.step(self.payload(200000))
        deadline = time.monotonic() + 2
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.01)
        runtime.llm.respond.assert_called_once()

        call = runtime.llm.respond.call_args
        self.assertEqual(call.args[0], '')  # nothing was said to answer
        trigger = call.kwargs['perception']['current_speaker']['attention_trigger']
        self.assertEqual(trigger['engagement_state'], 'affect_event')
        self.assertEqual(trigger['detail']['samples'], 18)

        prompt = runtime.state['current_turn']['model_context']
        self.assertEqual(prompt['attention_trigger']['engagement_state'], 'affect_event')
        system = call.kwargs.get('system') or call.args[1] if len(call.args) > 1 else ''
        if system:
            self.assertIn('不是情绪或健康判断', system)
            self.assertIn('不要断言他的心情', system)

    def test_a_suppressed_candidate_is_not_acted_on(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.llm = Mock()
        self.addCleanup(runtime.stop)
        payload = self.payload(200000)
        payload['workspace_events'][0]['admitted'] = False
        payload['workspace_events'] = []  # the host only forwards admitted ones
        with patch('brain.runtime.now_ms', return_value=200000):
            runtime.step(payload)
        self.assertFalse(runtime.jobs)
        runtime.llm.respond.assert_not_called()


class DoubleOpeningTests(unittest.TestCase):
    """Cross-session familiarity made a returning owner reach INVITED as well as
    the owner rule, so the same arrival could be opened on twice."""

    def test_only_attention_opens_on_an_arrival(self):
        """There used to be two rules that could open on the same arrival — the
        owner rule in Brain and the invitation in attention — and they did."""
        from brain.attention_trigger import AttentionTrigger
        trigger = AttentionTrigger()
        arrival = {'stamp_ms': 60000,
                   'attention': {'stamp_ms': 60000, 'engagement': {'state': 'IDLE', 'person_id': None}},
                   'workspace_events': [{'candidate_id': 'arrival:天行:60000', 'kind': 'arrival_event',
                                         'person_id': '天行', 'share': .4,
                                         'reasons': ['returned_after_absence'], 'summary': '回来了'}]}

        first = trigger.update(arrival, 60000, participants=(), last_heard_ms=40000)
        second = trigger.update(dict(arrival, stamp_ms=61000), 61000, participants=(), last_heard_ms=40000)

        self.assertIsNotNone(first)
        self.assertEqual(first['engagement_state'], 'arrival_event')
        self.assertIsNone(second)


class PriorityTests(unittest.TestCase):
    """What somebody just said outranks anything the robot thought of by itself.

    Field trace: attention reached the owner, the robot asked about something he
    had said a minute earlier, and in the same breath answered his "喂". Two
    replies to two different things.
    """

    def workspace(self, stamp, kind='memory_event', person='天行'):
        return {'stamp_ms': stamp,
                'attention': {'stamp_ms': stamp, 'engagement': {'state': 'IDLE', 'person_id': None}},
                'workspace_events': [{'candidate_id': f'{kind}:{person}:{stamp}', 'kind': kind,
                                      'person_id': person, 'share': .4, 'reasons': ['unfinished'],
                                      'summary': '之前问过没答'}]}

    def test_an_internal_candidate_waits_while_somebody_is_being_answered(self):
        from brain.attention_trigger import AttentionTrigger
        trigger = AttentionTrigger()

        self.assertIsNone(trigger.update(self.workspace(60000), 60000, participants=('天行',),
                                         last_heard_ms=10000, turn_in_flight=True))
        self.assertEqual(trigger.status()['reason'], 'live_turn_outranks_internal_candidate')

    def test_an_internal_candidate_waits_after_we_just_answered_somebody(self):
        from brain.attention_trigger import AttentionTrigger
        trigger = AttentionTrigger()
        trigger.note_reply(58000)

        self.assertIsNone(trigger.update(self.workspace(60000), 60000, participants=('天行',),
                                         last_heard_ms=40000))
        # 内生候选有更长的冷却（8 秒），邀请是 5 秒
        self.assertEqual(trigger.status()['reason'], 'just_spoke_internal_candidate_waits')

    def test_and_speaks_once_the_room_has_been_quiet(self):
        from brain.attention_trigger import AttentionTrigger
        trigger = AttentionTrigger()

        fired = trigger.update(self.workspace(60000), 60000, participants=('天行',), last_heard_ms=40000)

        self.assertIsNotNone(fired)
        self.assertEqual(fired['engagement_state'], 'memory_event')


class ReplyQueueTests(unittest.TestCase):
    """One reply per thing said, and stale ones are dropped rather than spoken.

    Field trace 13:57: the reply to "我感觉今天很热" arrived as a weather report
    left over from the previous question, and six seconds later a second opening
    followed about the heat. Two replies, one of them answering something nobody
    had just asked.
    """

    def invited(self, stamp, person='天行'):
        return {'stamp_ms': stamp,
                'attention': {'stamp_ms': stamp,
                              'engagement': {'state': 'INVITED', 'person_id': person,
                                             'confidence': .8, 'reasons': ['silent_invitation']}}}

    def test_an_invitation_right_after_we_spoke_is_dropped(self):
        from brain.attention_trigger import AttentionTrigger
        trigger = AttentionTrigger()
        trigger.note_reply(60000)

        self.assertIsNone(trigger.update(self.invited(63000), 63000, participants=('天行',), last_heard_ms=59000))
        self.assertEqual(trigger.status()['reason'], 'just_spoke_nothing_new_to_add')

    def test_and_allowed_once_the_silence_has_lasted(self):
        from brain.attention_trigger import AttentionTrigger
        trigger = AttentionTrigger()
        trigger.note_reply(60000)

        fired = trigger.update(self.invited(66000), 66000, participants=('天行',), last_heard_ms=65000)

        self.assertIsNotNone(fired)


class HeldUtteranceTests(unittest.TestCase):
    """Not sure it was for us: write it down, say nothing, answer if attention comes.

    The gate was made more sensitive and then everything sounded like it was
    addressed to the robot. The middle ground is two bars: answer what is
    certain, keep what is merely likely, and let attention decide whether the
    kept sentence ever gets a reply.
    """

    def runtime(self):
        from brain.runtime import BrainRuntime
        from unittest.mock import Mock
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '嗯呀'
        runtime.memory.search = Mock(return_value=[])
        self.addCleanup(runtime.stop)
        return runtime

    def held_turn(self, stamp, text='这个方案行吗'):
        return {'utterance_id': 'u-held', 'person_id': '天行', 'stamp_ms': stamp,
                'transcript': {'text': text, 'is_final': True, 'emitted_ms': stamp, 'ended_ms': stamp},
                'attention': {'listen': False, 'addressed_to_robot': False, 'confidence': .0,
                              'held_for_attention': True, 'reasons': ['held_pending_attention']},
                'held_for_attention': True,
                'addressee_detail': {'score': .6, 'decision': 'hold', 'terms': {'gaze_while_speaking': .6}}}

    def test_a_held_sentence_is_not_answered_when_it_is_heard(self):
        runtime = self.runtime()
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step({'utterances': [self.held_turn(100000)]})
        time.sleep(.2)
        runtime.tts.speak.assert_not_called()
        self.assertEqual([item['text'] for item in runtime.held], ['这个方案行吗'])

    def test_and_is_answered_from_when_attention_arrives(self):
        runtime = self.runtime()
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step({'utterances': [self.held_turn(100000)]})
        state = {'stamp_ms': 104000,
                 'attention': {'stamp_ms': 104000,
                               'engagement': {'state': 'INVITED', 'person_id': '天行',
                                              'confidence': .8, 'reasons': ['silent_invitation']}}}
        with patch('brain.runtime.now_ms', return_value=104000):
            runtime.step(state)
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)

        self.assertEqual(runtime.llm.respond.call_args.args[0], '这个方案行吗')
        self.assertFalse(list(runtime.held))


class VoiceBoundHeldUtteranceTests(HeldUtteranceTests):
    """人群在镜头外说话，一个人回头：他刚才那句话得由他的脸来认领。

    Off camera there is no face, so the sentence is filed under the voice
    cluster. Attention then arrives on a face. Perception says which voices
    belong to that face; without using it the robot turns to the person and
    opens its mouth with nothing to say.
    """

    def test_a_sentence_held_under_a_voice_is_answered_when_its_speaker_turns_round(self):
        runtime = self.runtime()
        crowd = dict(self.held_turn(100000, '这个能便宜点吗'), person_id='stranger_dd')
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step({'utterances': [crowd]})
        state = {'stamp_ms': 104000,
                 'voice_bindings': {'guest_2': ['stranger_dd']},
                 'attention': {'stamp_ms': 104000,
                               'engagement': {'state': 'INVITED', 'person_id': 'guest_2',
                                              'confidence': .8, 'reasons': ['silent_invitation']}}}
        with patch('brain.runtime.now_ms', return_value=104000):
            runtime.step(state)
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)

        self.assertEqual(runtime.llm.respond.call_args.args[0], '这个能便宜点吗')
        self.assertFalse(list(runtime.held))

    def test_without_a_binding_somebody_elses_sentence_is_not_claimed(self):
        runtime = self.runtime()
        crowd = dict(self.held_turn(100000, '这个能便宜点吗'), person_id='stranger_dd')
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step({'utterances': [crowd]})
        state = {'stamp_ms': 104000,
                 'attention': {'stamp_ms': 104000,
                               'engagement': {'state': 'INVITED', 'person_id': 'guest_2',
                                              'confidence': .8, 'reasons': ['silent_invitation']}}}
        with patch('brain.runtime.now_ms', return_value=104000):
            runtime.step(state)
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)

        self.assertNotEqual(runtime.llm.respond.call_args and runtime.llm.respond.call_args.args[0], '这个能便宜点吗')
        self.assertEqual([item['text'] for item in runtime.held], ['这个能便宜点吗'])


class OneQuestionOneReplyTests(unittest.TestCase):
    """一个问题一个回话：不重复回答，也不把自己的半句话切断。

    From the field log, one question produced three replies:

      17:55:29 天行「你叫什么名字？」
      17:55:31 robot plays「我是小圆呀」
      17:55:32 heard「我是小圆样。」(self_echo 0.457, filed under a stranger)
      17:55:33 robot answers that, is heard again, answers again…

    Each of those also arrived as "new addressed speech", which cut the reply
    that was already playing — which is where the half-finished replies and the
    constant interruptions came from. Same bug, three symptoms.
    """

    def runtime(self):
        from brain.runtime import BrainRuntime
        from unittest.mock import Mock
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '我是小圆呀'
        runtime.memory.search = Mock(return_value=[])
        self.addCleanup(runtime.stop)
        return runtime

    def granted(self, text, stamp, person='天行', key=None):
        return {'utterance_id': key or ('u-' + str(stamp)), 'person_id': person, 'stamp_ms': stamp,
                'transcript': {'text': text, 'is_final': True, 'emitted_ms': stamp, 'ended_ms': stamp},
                'attention': {'listen': True, 'addressed_to_robot': True, 'confidence': .9,
                              'reasons': ['directed_call']},
                'addressee': 'robot'}

    def test_our_own_reply_heard_back_is_not_a_question(self):
        runtime = self.runtime()
        runtime.state['events'] = [{'text': '我是小圆呀', 'stamp_ms': 100000}]
        with patch('brain.runtime.now_ms', return_value=101000):
            runtime.step({'utterances': [self.granted('我是小圆样。', 101000, person='stranger_a4b2')]})
        time.sleep(.2)
        runtime.llm.respond.assert_not_called()
        self.assertTrue(any(item['kind'] == 'own_reply_heard_again' for item in runtime.state['timeline']))

    def test_a_person_actually_asking_is_still_answered(self):
        runtime = self.runtime()
        runtime.state['events'] = [{'text': '我是小圆呀', 'stamp_ms': 100000}]
        with patch('brain.runtime.now_ms', return_value=101000):
            runtime.step({'utterances': [self.granted('那你会做什么呀', 101000)]})
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(runtime.llm.respond.call_args.args[0], '那你会做什么呀')

    def test_a_second_sentence_before_we_speak_replaces_the_first(self):
        runtime = self.runtime()
        runtime.state['s2s'].update(thinking=True)
        runtime.state['current_turn'] = {'turn_id': 'u-1', 'person_id': '天行', 'user_text': '那个方案',
                                         'received_ms': 100000, 'playback_started_ms': None}
        with patch('brain.runtime.now_ms', return_value=101200):
            runtime.step({'utterances': [self.granted('你觉得行吗', 101200, key='u-2')]})
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(runtime.llm.respond.call_args.args[0], '你觉得行吗')
        self.assertEqual(runtime.state['s2s']['interruptions'], 0)

    def test_but_a_sentence_that_arrives_mid_playback_waits_its_turn(self):
        runtime = self.runtime()
        runtime.state['s2s'].update(speaking=True)
        runtime.state['current_turn'] = {'turn_id': 'u-1', 'person_id': '天行', 'user_text': '那个方案',
                                         'received_ms': 100000, 'playback_started_ms': 100800}
        with patch('brain.runtime.now_ms', return_value=101200):
            runtime.step({'utterances': [self.granted('你觉得行吗', 101200, key='u-2')]})
        time.sleep(.2)
        runtime.llm.respond.assert_not_called()
        self.assertEqual(runtime.pending_user_turn['utterance_id'], 'u-2')
        self.assertTrue(any(item['kind'] == 'user_turn_deferred' for item in runtime.state['timeline']))

    def test_somebody_else_cutting_in_is_still_an_interruption(self):
        runtime = self.runtime()
        runtime.state['s2s'].update(speaking=True)
        runtime.state['current_turn'] = {'turn_id': 'u-1', 'person_id': '天行', 'user_text': '那个方案',
                                         'received_ms': 100000, 'playback_started_ms': 100800}
        with patch('brain.runtime.now_ms', return_value=101200):
            runtime.step({'utterances': [self.granted('等一下', 101200, person='guest_2', key='u-3')]})
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(runtime.llm.respond.call_args.args[0], '等一下')


class NothingToSayTests(HeldUtteranceTests):
    """注意力被唤起 ≠ 要说话。

    In the field the robot looked up at somebody it had already finished
    talking to, was handed the old context, and answered a sentence from
    minutes earlier. Attention says where to look; whether there is anything to
    say is a separate question, and the answer is usually no.
    """

    def invited(self, stamp, person='天行'):
        return {'stamp_ms': stamp,
                'attention': {'stamp_ms': stamp,
                              'engagement': {'state': 'INVITED', 'person_id': person,
                                             'confidence': .8, 'reasons': ['silent_invitation']}}}

    def heard(self, text, stamp, person='天行'):
        """The shape _remember_heard stores, so the reply path can read it."""
        return {'utterance_id': 'h-' + str(stamp), 'person_id': person, 'stamp_ms': stamp,
                'transcript': {'text': text, 'is_final': True, 'emitted_ms': stamp, 'ended_ms': stamp},
                'attention': {'addressed_to_robot': True}}

    def test_a_closed_conversation_gets_a_look_and_no_words(self):
        runtime = self.runtime()
        runtime.state['heard'] = [self.heard('你好', 90000)]
        runtime.state['events'] = [{'person_id': '天行', 'stamp_ms': 91000, 'text': '你好呀'}]
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step(self.invited(100000))
        time.sleep(.2)
        runtime.llm.respond.assert_not_called()
        self.assertTrue(any(item['kind'] == 'attention_without_anything_to_say'
                            for item in runtime.state['timeline']))

    def test_but_a_sentence_of_theirs_we_never_answered_is_taken_up(self):
        runtime = self.runtime()
        runtime.state['heard'] = [self.heard('这个能便宜点吗', 99000)]
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step(self.invited(100000))
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)
        runtime.llm.respond.assert_called()

    def test_and_a_candidate_that_arrived_with_its_own_content_still_speaks(self):
        runtime = self.runtime()
        runtime.state['heard'] = [self.heard('你好', 90000)]
        runtime.state['events'] = [{'person_id': '天行', 'stamp_ms': 91000, 'text': '你好呀'}]
        state = {'stamp_ms': 100000, 'attention': {'stamp_ms': 100000, 'engagement': {}},
                 'workspace_events': [{'kind': 'arrival_event', 'person_id': '天行', 'share': .5,
                                       'summary': '认识的人回来了，房间是安静的',
                                       'detail': {'quiet_ms': 60000}, 'reasons': ['returned_after_absence']}]}
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step(state)
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)
        runtime.llm.respond.assert_called()

    def test_a_question_of_ours_they_never_answered_is_a_reason_to_speak(self):
        runtime = self.runtime()
        runtime.state['heard'] = [self.heard('你好', 90000)]
        runtime.state['events'] = [{'person_id': '天行', 'stamp_ms': 91000, 'text': '你好呀'}]
        runtime.expected_answer.offered('t-1', '天行', '你吃了吗', 95000, True, {})
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step(self.invited(100000))
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)
        runtime.llm.respond.assert_called()


class WhatToSayFirstTests(HeldUtteranceTests):
    """自言自语之后转向它：先接他刚说的话，而不是"好久不见"。

    From the field, 11:03: he said 我今天身体不太舒服。and 想出去玩。to himself,
    then turned round. Attention woke, and the robot greeted him as somebody
    who had just appeared — true, and the worse of the two answers. Both
    candidates were real; the order was wrong.
    """

    def invited(self, stamp, person='天行'):
        return {'stamp_ms': stamp,
                'attention': {'stamp_ms': stamp,
                              'engagement': {'state': 'INVITED', 'person_id': person,
                                             'confidence': .8, 'reasons': ['silent_invitation']}}}

    def arrival(self, stamp, person='天行'):
        return {'stamp_ms': stamp, 'attention': {'stamp_ms': stamp, 'engagement': {}},
                'workspace_events': [{'kind': 'arrival_event', 'person_id': person, 'share': .5,
                                      'summary': '认识的人回来了，房间是安静的',
                                      'detail': {'quiet_ms': 60000},
                                      'reasons': ['returned_after_absence']}]}

    def heard(self, text, stamp, person='天行'):
        return {'utterance_id': 'h-' + str(stamp), 'person_id': person, 'stamp_ms': stamp,
                'transcript': {'text': text, 'is_final': True, 'emitted_ms': stamp, 'ended_ms': stamp},
                'attention': {'addressed_to_robot': False}}

    def run_step(self, runtime, state, stamp):
        with patch('brain.runtime.now_ms', return_value=stamp):
            runtime.step(state)
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)

    def test_what_he_said_to_himself_outranks_a_welcome_back(self):
        runtime = self.runtime()
        runtime.state['heard'] = [self.heard('想出去玩', 99000), self.heard('我今天身体不太舒服', 98000)]
        self.run_step(runtime, self.arrival(100000), 100000)
        self.assertEqual(runtime.llm.respond.call_args.args[0], '想出去玩')
        self.assertEqual(runtime.state['current_turn']['speak_because'], 'their_recent_words')

    def test_a_kept_sentence_outranks_even_that(self):
        runtime = self.runtime()
        with patch('brain.runtime.now_ms', return_value=99000):
            runtime.step({'utterances': [self.held_turn(99000, '这个方案行吗')]})
        runtime.state['heard'].append(self.heard('随便说的一句', 99500))
        self.run_step(runtime, self.invited(100000), 100000)
        self.assertEqual(runtime.llm.respond.call_args.args[0], '这个方案行吗')
        self.assertEqual(runtime.state['current_turn']['speak_because'], 'held_sentence')

    def test_words_from_ten_minutes_ago_do_not_outrank_it(self):
        runtime = self.runtime()
        runtime.state['heard'] = [self.heard('想出去玩', 100000 - 600000)]
        self.run_step(runtime, self.arrival(100000), 100000)
        self.assertEqual(runtime.llm.respond.call_args.args[0], '')
        self.assertEqual(runtime.state['current_turn']['speak_because'], 'workspace_candidate')

    def test_and_something_we_already_answered_is_not_answered_again(self):
        runtime = self.runtime()
        runtime.state['heard'] = [self.heard('想出去玩', 99000)]
        runtime.state['events'] = [{'person_id': '天行', 'stamp_ms': 99500, 'text': '好呀'}]
        self.run_step(runtime, self.invited(100000), 100000)
        runtime.llm.respond.assert_not_called()
        self.assertTrue(any(item['kind'] == 'attention_without_anything_to_say'
                            for item in runtime.state['timeline']))


class AnswerIsNotAGreetingTests(HeldUtteranceTests):
    """答他刚说的那句，不是"延后主动问候"。

    11:42 in the field: he said 我想出去玩了。while looking at the robot, the
    reply 好的，你想去哪玩? was produced — and cancelled as
    user_speech_preempts_proactive, marked 被打断. Two mistakes in one line.
    A turn that answers what this person just said is not a greeting waiting
    for a gap; and raw VAD is not evidence that a person is talking, because it
    goes active on the robot's own playback.
    """

    def speaking_now(self):
        return {'stamp_ms': 100000,
                'voice': {'last_vad': {'active': True, 'stamp_ms': 100000},
                          'playback': {'speaking': False}}}

    def test_their_own_voice_does_not_cancel_the_answer_to_their_sentence(self):
        runtime = self.runtime()
        runtime.state['current_turn'] = {'speak_because': 'their_recent_words', 'person_id': '天行',
                                         'turn_id': 'answer', 'attention_trigger': {'kind': 'attention_engagement'}}
        runtime.state['s2s']['thinking'] = True
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step(self.speaking_now())
        self.assertFalse(runtime.cancel.is_set())

    def test_but_a_greeting_still_gives_way_to_somebody_talking(self):
        runtime = self.runtime()
        runtime.state['current_turn'] = {'speak_because': 'workspace_candidate', 'person_id': '天行',
                                         'turn_id': 'greeting'}
        runtime.state['s2s']['thinking'] = True
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step(self.speaking_now())
        self.assertTrue(runtime.cancel.is_set())
        self.assertTrue(any(item.get('reason') == 'user_speech_preempts_proactive'
                            for item in runtime.state['timeline']))

    def test_our_own_playback_is_not_somebody_talking(self):
        runtime = self.runtime()
        runtime.state['current_turn'] = {'speak_because': 'workspace_candidate', 'person_id': '天行',
                                         'turn_id': 'greeting'}
        runtime.state['s2s']['thinking'] = True
        state = {'stamp_ms': 100000,
                 'voice': {'last_vad': {'active': True, 'stamp_ms': 100000},
                           'playback': {'speaking': True}}}
        with patch('brain.runtime.now_ms', return_value=100000):
            runtime.step(state)
        self.assertFalse(runtime.cancel.is_set())


class ItsOwnTurnsAreContextTests(HeldUtteranceTests):
    """它自己说过的话也是上下文，否则它会鹦鹉学舌。

    12:30 in the field: 「哎，我觉得上迪士尼乐园挺好玩的。」 was answered with
    「上海迪士尼挺好玩的。」, and 「你觉得新开的那家冰淇淋？」 with
    「冰淇淋挺好吃的。」 — both parroting, because recent_conversation was built
    from heard utterances only. As far as the model could tell, nothing had
    been said back yet. The prompt tells it that a line from 小圆 is its own;
    there were none in the context to see.
    """

    def granted(self, text, stamp, person='天行', key=None):
        return {'utterance_id': key or ('u-' + str(stamp)), 'person_id': person, 'stamp_ms': stamp,
                'transcript': {'text': text, 'is_final': True, 'emitted_ms': stamp, 'ended_ms': stamp},
                'attention': {'listen': True, 'addressed_to_robot': True, 'confidence': .9,
                              'reasons': ['directed_call']},
                'addressee': 'robot'}

    def test_the_model_sees_both_sides_in_order(self):
        runtime = self.runtime()
        runtime.state['events'] = [{'text': '迪士尼人多吗', 'stamp_ms': 100500, 'person_id': '天行'}]
        runtime.state['heard'] = [{'utterance_id': 'h1', 'person_id': '天行', 'stamp_ms': 100000,
                                   'transcript': {'text': '上海迪士尼挺好玩的', 'emitted_ms': 100000,
                                                  'ended_ms': 100000},
                                   'attention': {'addressed_to_robot': False}}]
        with patch('brain.runtime.now_ms', return_value=101000):
            runtime.step({'utterances': [self.granted('你觉得呢', 101000, key='u-now')]})
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)

        call = runtime.llm.respond.call_args
        self.assertEqual(call.args[0], '你觉得呢')
        recent = call.kwargs['perception']['current_speaker']['recent_conversation']
        self.assertEqual([(row['person_id'], row['text']) for row in recent],
                         [('天行', '上海迪士尼挺好玩的'), ('小圆', '迪士尼人多吗')])

    def test_our_own_reply_is_not_offered_as_the_question(self):
        """The loop that collects them used the name `text`, which already held
        the sentence being answered; shadowing it sent our last reply to the
        model as the user's question."""
        runtime = self.runtime()
        runtime.state['events'] = [{'text': '我是小圆呀', 'stamp_ms': 100000}]
        with patch('brain.runtime.now_ms', return_value=101000):
            runtime.step({'utterances': [self.granted('那你会做什么呀', 101000)]})
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(runtime.llm.respond.call_args.args[0], '那你会做什么呀')

    def test_old_replies_do_not_follow_the_conversation_around(self):
        runtime = self.runtime()
        runtime.state['events'] = [{'text': '很久以前说的', 'stamp_ms': 101000 - 300000}]
        with patch('brain.runtime.now_ms', return_value=101000):
            runtime.step({'utterances': [self.granted('那个方案你觉得怎么样', 101000)]})
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.02)
        recent = runtime.llm.respond.call_args.kwargs['perception']['current_speaker']['recent_conversation']
        self.assertEqual([row['text'] for row in recent], [])
