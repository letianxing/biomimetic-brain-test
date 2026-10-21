import json
import time
import unittest
from unittest.mock import Mock, patch
from brain.runtime import BrainRuntime


class ReflexReplyTests(unittest.TestCase):
    def setUp(self):
        self.runtime=BrainRuntime(memory_path=':memory:')
        self.runtime.tts=Mock(enabled=True)
        self.runtime.tts.speak.return_value=True
        self.runtime.llm=Mock()
        self.runtime.memory.search=Mock(return_value=[])
        self.addCleanup(self.runtime.stop)

    def wait(self):
        end=time.monotonic()+2
        while self.runtime.jobs and time.monotonic()<end: time.sleep(.01)
        self.assertFalse(self.runtime.jobs)

    def test_startle_short_exclamation_once_without_llm(self):
        stamp=int(time.time()*1000)
        focus={'id':'sound:1','kind':'startle','active':True,'source':'audio','stamp_ms':stamp,'target_id':'sound:1','listen':False}
        self.runtime.step({'reflex_attention':focus})
        self.wait()
        self.runtime.step({'reflex_attention':focus})
        self.runtime.step({'reflex_attention':dict(focus,id='sound:2')})
        self.runtime.tts.speak.assert_called_once()
        self.assertEqual(self.runtime.tts.speak.call_args.args[0],'哎呀！')
        self.runtime.llm.respond.assert_not_called()
        self.runtime.memory.search.assert_not_called()
        self.assertEqual(self.runtime.state['current_turn']['path'],'reflex')

    def test_stale_startle_and_robot_echo_never_reply(self):
        self.runtime.step({'reflex_attention':{'active':True,'kind':'startle','id':'old','stamp_ms':1},'utterances':[{'utterance_id':'e','source':'robot_echo','stamp_ms':1,'transcript':{'text':'机器人回声'}}]})
        self.runtime.tts.speak.assert_not_called()
        records=[json.loads(row[0]) for row in self.runtime.memory.db.execute('SELECT payload FROM outbox')]
        self.assertTrue(any(r['kind']=='robot_echo' for r in records))
        self.assertFalse(any(r['kind']=='heard_utterance' for r in records))

    def test_an_identity_dip_does_not_produce_a_second_opening(self):
        self.runtime._submit=Mock()
        owner={'resolved_person_id':'owner','role':'owner','identity_confidence':.95,'active':True}
        dipped=dict(owner,identity_confidence=.4)
        for stamp,person in [(100000,owner),(106000,owner),(106500,dipped),(107000,owner),(112000,owner)]:
            with patch('brain.runtime.now_ms',return_value=stamp):
                self.runtime.step({'person_hypotheses':[person],'attention':{'person_id':'owner','listen':True,'confidence':.9}})
        self.runtime._submit.assert_not_called()


    def test_old_playback_callback_cannot_override_new_turn(self):
        self.runtime.generation=2
        self.runtime.state['s2s']['speaking']=True
        self.runtime._turn_playback(1,{'turn_id':'old','person_id':'person1','playback_started_ms':1,'assistant_text':'旧回复'},False)
        self.assertTrue(self.runtime.state['s2s']['speaking'])

    def test_recall_echo_filter_requires_playback_evidence(self):
        from brain.memory_echo import is_playback_echo
        text='我们继续聊之前公园的话题'
        heard={'kind':'heard_utterance','text':text,'structured':{'transcript':{'started_ms':1000,'ended_ms':2000}}}
        reply={'kind':'dialogue_turn','text':text,'structured':{'timing':{'playback_started_ms':900,'playback_ended_ms':2100}}}
        self.assertTrue(is_playback_echo(heard,[reply]))
        self.assertFalse(is_playback_echo(heard,[]))
        reply['structured']['timing']['playback_ended_ms']=100
        self.assertFalse(is_playback_echo(heard,[reply]))

    def test_registered_owner_teasing_requires_identity_evidence(self):
        turn={'person_id':'小天','transcript':{'text':'我是你的主人','speaker':{'speaker_id':'小天','speaker_role':'owner','similarity':.9}}}
        self.assertIn('不要来挑逗',self.runtime._registration_reply(turn).text)
        turn['transcript']['speaker']['similarity']=.2
        self.assertIsNone(self.runtime._registration_reply(turn))
        turn['transcript']['speaker']['similarity']=.9
        turn['transcript']['text']='我喜欢去公园散步'
        self.assertIsNone(self.runtime._registration_reply(turn))
        turn['transcript']['repeats_enrollment']=True
        self.assertEqual(self.runtime._registration_reply(turn).reason,'already_registered')

    def test_current_speaker_memories_and_environment_both_reach_llm(self):
        def search(query='',person=''):
            return [{'trace_id':'personal','text':'小天喜欢公园','entity':{'person_id':'stranger_old'}}] if person else [{'trace_id':'ambient','text':'下午三点见','entity':{'person_id':'person2'}}]
        self.runtime.memory.search.side_effect=search
        self.runtime.llm.respond.return_value='你们约了下午三点去公园。'
        stamp=int(time.time()*1000)
        self.runtime.step({'utterances':[{'utterance_id':'question','person_id':'小天','stamp_ms':stamp,'transcript':{'text':'我们之前约了几点','is_final':True,'emitted_ms':stamp,'speaker':{'speaker_id':'小天','aliases':['stranger_old']}},'attention':{'listen':True,'addressed_to_robot':True,'confidence':.9},'evidence':{'vision':[{'person_id':'小天','face_confidence':.95}]}}]})
        self.wait()
        call=self.runtime.llm.respond.call_args
        self.assertEqual(call.args[0],'我们之前约了几点')
        self.assertEqual({m['trace_id'] for m in call.args[1]},{'personal','ambient'})
        self.assertEqual(call.kwargs['perception']['current_speaker']['person_id'],'小天')

    def test_bystander_final_delays_owner_greeting(self):
        stamp=int(time.time()*1000)
        self.runtime.owners={'owner':{'since':stamp-10000,'last_seen':stamp,'greeted':False,'last_greeting':0}}
        self.runtime.step({'person_hypotheses':[{'resolved_person_id':'owner','role':'owner','identity_confidence':.9,'active':True}], 'utterances':[{'utterance_id':'ambient','person_id':'stranger_a','stamp_ms':stamp,'transcript':{'text':'咱们下午去公园','is_final':True,'emitted_ms':stamp},'attention':{'listen':False,'addressed_to_robot':False}}]})
        self.runtime.tts.speak.assert_not_called()
        self.assertEqual(len(self.runtime.state['heard']),1)

    def test_owner_must_be_locked_before_proactive_and_loss_cancels(self):
        owner={'resolved_person_id':'owner','role':'owner','identity_confidence':.9,'active':True}
        self.runtime._submit=Mock()
        for stamp in [100000,101000,102000]:
            with patch('brain.runtime.now_ms',return_value=stamp):self.runtime.step({'person_hypotheses':[owner],'attention':{'person_id':'other','listen':True,'confidence':.9}})
        self.runtime._submit.assert_not_called()
        for stamp in [103000,108000]:
            with patch('brain.runtime.now_ms',return_value=stamp):self.runtime.step({'person_hypotheses':[owner],'attention':{'person_id':'owner','target_id':'person:owner','listen':True,'confidence':.9}})
        # The owner rule tracks the lock and the wait; it does not open a turn.
        # Somebody familiar coming back is an attention candidate (arrival_event)
        # so it can be weighed against everything else going on.
        self.runtime._submit.assert_not_called()
        initiative=self.runtime.state['initiative']['owner']
        self.assertTrue(initiative['locked'])
        self.assertEqual(initiative['wait_remaining_ms'],0)
        self.runtime.state['current_turn']={'proactive':True,'person_id':'owner','turn_id':'greeting'}
        self.runtime.state['s2s']['thinking']=True
        # One cycle is attention passing through a reflex, not the person
        # leaving: ORIENTING and ACQUIRING move the target off the person every
        # couple of seconds by design, and cancelling on that cut replies in
        # half all through the field session.
        elsewhere={'attention':{'person_id':'other','listen':True,'confidence':.9}}
        with patch('brain.runtime.now_ms',return_value=109000):self.runtime.step(elsewhere)
        self.assertFalse(self.runtime.cancel.is_set())
        with patch('brain.runtime.now_ms',return_value=109700):self.runtime.step(elsewhere)
        self.assertTrue(self.runtime.cancel.is_set())
        self.assertEqual(self.runtime.state['s2s']['last_interruption']['reason'],'owner_attention_lock_lost')

    def test_a_reply_to_what_they_said_is_not_cancelled_by_the_camera(self):
        """说了就说完：答他刚说的那句，不需要摄像头一直锁着他。"""
        self.runtime.state['current_turn']={'speak_because':'their_recent_words','person_id':'owner',
                                            'turn_id':'answer','attention_trigger':{'kind':'attention_engagement'}}
        self.runtime.state['s2s']['thinking']=True
        elsewhere={'attention':{'person_id':'other','listen':True,'confidence':.9}}
        for stamp in (109000,109700,110400,111100):
            with patch('brain.runtime.now_ms',return_value=stamp):self.runtime.step(elsewhere)
        self.assertFalse(self.runtime.cancel.is_set())

    def test_nor_is_a_greeting_that_has_already_started_speaking(self):
        self.runtime.state['current_turn']={'proactive':True,'person_id':'owner','turn_id':'greeting',
                                            'playback_started_ms':108500}
        self.runtime.state['s2s'].update(thinking=False,speaking=True)
        elsewhere={'attention':{'person_id':'other','listen':True,'confidence':.9}}
        for stamp in (109000,109700,110400):
            with patch('brain.runtime.now_ms',return_value=stamp):self.runtime.step(elsewhere)
        self.assertFalse(self.runtime.cancel.is_set())

    def test_the_owner_rule_never_opens_a_turn_of_its_own(self):
        """It used to submit a turn whose user text was an instruction to itself
        ("主人刚刚出现，请结合之前的话题自然搭话"), which the model then read out."""
        owner={'resolved_person_id':'owner','role':'owner','identity_confidence':.9,'active':True}
        for stamp in [100000,105000,110000]:
            with patch('brain.runtime.now_ms',return_value=stamp):
                self.runtime.step({'person_hypotheses':[owner],'attention':{'person_id':'owner','listen':True,'confidence':.9}})
        self.wait()
        self.runtime.tts.speak.assert_not_called()


    def test_owner_turns_from_bystander_chat_and_asks_agreement_no_greeting(self):
        stamp=int(time.time()*1000)
        owner={'resolved_person_id':'owner','role':'owner','identity_confidence':.95,'active':True}
        def turn(key,text,when,person,addressed):
            return {'utterance_id':key,'person_id':person,'stamp_ms':when,'transcript':{'text':text,'is_final':True,'ended_ms':when,'emitted_ms':when},'attention':{'listen':addressed,'addressed_to_robot':addressed,'confidence':.9}}
        self.runtime.step({'utterances':[turn('p1','下午三点去公园比较合适',stamp-1500,'owner',False),turn('p2','那时太阳不太晒',stamp-1000,'person2',False)]})
        self.runtime.llm.respond.return_value='是的，按你们刚才的安排，下午三点去公园比较合适。'
        focused={'person_id':'owner','target_id':'person:owner','listen':True,'confidence':.9}
        self.runtime.step({'person_hypotheses':[owner],'attention':focused,'utterances':[turn('ask','你说是吧',stamp,'owner',True)]})
        self.wait()
        request=self.runtime.llm.respond.call_args
        self.assertIsNone(request.kwargs['proactive'])
        recent=request.kwargs['perception']['current_speaker']['recent_conversation']
        self.assertEqual([r['text'] for r in recent],['下午三点去公园比较合适','那时太阳不太晒'])
        self.assertEqual(request.args[0],'你说是吧')
        with patch('brain.runtime.now_ms',return_value=stamp+10000):self.runtime.step({'person_hypotheses':[owner],'attention':focused})
        self.runtime.llm.respond.assert_called_once()

    def test_partial_speech_cancels_pending_greeting_before_final_asr(self):
        self.runtime.state['current_turn']={'turn_id':'greeting','person_id':'owner','proactive':True}
        self.runtime.state['s2s']['thinking']=True
        stamp=int(time.time()*1000)
        self.runtime.step({'attention':{'person_id':'owner','listen':True,'confidence':.9},'voice':{'last_streaming_transcript':{'text':'你说','emitted_ms':stamp,'source':'microphone','is_final':False}}})
        self.assertTrue(self.runtime.cancel.is_set())
        self.assertEqual(self.runtime.state['s2s']['last_deferral']['reason'],'user_speech_preempts_proactive')

    def test_silence_count_starts_at_lock_and_speech_restarts_wait(self):
        self.runtime._submit=Mock()
        owner={'resolved_person_id':'owner','role':'owner','identity_confidence':.95,'active':True}
        focus={'person_id':'owner','listen':True,'confidence':.9}
        def step(stamp,speaking=False,target=None):
            with patch('brain.runtime.now_ms',return_value=stamp):
                self.runtime.step({'person_hypotheses':[owner],'attention':focus if target is None else {'person_id':target,'listen':True,'confidence':.9},'voice':{'last_vad':{'active':speaking,'stamp_ms':stamp}}})
        step(100000);step(104000)
        self.runtime._submit.assert_not_called()
        step(104500,True);step(105000);step(109000)
        self.runtime._submit.assert_not_called()
        step(109500)
        # The owner rule tracks the lock and the wait; it does not open a turn.
        # Somebody familiar coming back is an attention candidate (arrival_event)
        # so it can be weighed against everything else going on.
        self.runtime._submit.assert_not_called()
        initiative=self.runtime.state['initiative']['owner']
        self.assertTrue(initiative['locked'])
        self.assertEqual(initiative['wait_remaining_ms'],0)

    def test_attention_switch_restarts_owner_wait(self):
        self.runtime._submit=Mock()
        owner={'resolved_person_id':'owner','role':'owner','identity_confidence':.95,'active':True}
        for stamp,person in [(100000,'owner'),(104000,'other'),(105000,'owner'),(109999,'owner')]:
            with patch('brain.runtime.now_ms',return_value=stamp):self.runtime.step({'person_hypotheses':[owner],'attention':{'person_id':person,'listen':True,'confidence':.9}})
        self.runtime._submit.assert_not_called()
        with patch('brain.runtime.now_ms',return_value=110000):self.runtime.step({'person_hypotheses':[owner],'attention':{'person_id':'owner','listen':True,'confidence':.9}})
        # The owner rule tracks the lock and the wait; it does not open a turn.
        # Somebody familiar coming back is an attention candidate (arrival_event)
        # so it can be weighed against everything else going on.
        self.runtime._submit.assert_not_called()
        initiative=self.runtime.state['initiative']['owner']
        self.assertTrue(initiative['locked'])
        self.assertEqual(initiative['wait_remaining_ms'],0)

    def test_startle_finishes_before_queued_user_question(self):
        import threading
        began=threading.Event();release=threading.Event()
        def speak(text,**kwargs):
            if text=='哎呀！':
                kwargs['on_playback'](True);began.set();release.wait(1);kwargs['on_playback'](False)
            return True
        self.runtime.tts.speak.side_effect=speak
        self.runtime.llm.respond.return_value='刚才约的是下午三点。'
        stamp=int(time.time()*1000)
        self.runtime.step({'reflex_attention':{'id':'bang','active':True,'kind':'startle','stamp_ms':stamp,'target_id':'sound:bang'}})
        self.assertTrue(began.wait(1))
        question={'utterance_id':'question','person_id':'p','stamp_ms':stamp,'transcript':{'text':'刚才约了几点','is_final':True,'emitted_ms':stamp},'attention':{'listen':True,'addressed_to_robot':True,'confidence':.9}}
        self.runtime.step({'utterances':[question]})
        self.assertIsNotNone(self.runtime.pending_user_turn)
        self.runtime.tts.interrupt.assert_not_called()
        release.set();self.wait()
        self.runtime.step({});self.wait()
        self.assertEqual([call.args[0] for call in self.runtime.tts.speak.call_args_list],['哎呀！','刚才约的是下午三点。'])

    def test_deferring_unspoken_greeting_is_not_counted_as_playback_interruption(self):
        stamp=int(time.time()*1000)
        self.runtime.state['current_turn']={'turn_id':'draft','person_id':'owner','proactive':True}
        self.runtime.state['s2s']['thinking']=True
        self.runtime.step({'attention':{'person_id':'owner','listen':True,'confidence':.9},'voice':{'last_vad':{'active':True,'stamp_ms':stamp}}})
        self.assertEqual(self.runtime.state['s2s']['interruptions'],0)
        self.assertEqual(self.runtime.state['s2s']['proactive_deferrals'],1)

    def test_polite_yield_waits_for_boundary_and_emergency_cancels_timer(self):
        self.runtime.state['s2s']['speaking']=True
        stamp=int(time.time()*1000)
        self.runtime.state['current_turn']={'turn_id':'old','person_id':'p','assistant_text':'第一句话。','playback_started_ms':stamp,'status':'speaking'}
        generation=self.runtime.generation
        self.runtime._request_yield('social',{}, {'fade_ms':40,'max_boundary_wait_ms':350})
        self.assertFalse(self.runtime.cancel.is_set());self.runtime.tts.interrupt.assert_not_called()
        self.runtime._turn_playback(generation,self.runtime.state['current_turn'].copy(),False)
        self.assertTrue(self.runtime.cancel.is_set())
        self.runtime.tts.interrupt.assert_called_once_with(fade_ms=40)
        self.assertFalse(self.runtime.state['s2s']['yield_pending'])
        self.runtime.state['s2s']['speaking']=True
        self.runtime.cancel=__import__('threading').Event()
        self.runtime._request_yield('social2',{}, {'fade_ms':40,'max_boundary_wait_ms':100})
        self.runtime._interrupt('emergency',fade_ms=5)
        time.sleep(.15)
        self.assertEqual(self.runtime.tts.interrupt.call_count,2)
        self.assertEqual(self.runtime.tts.interrupt.call_args.kwargs['fade_ms'],5)

    def test_a_brief_visual_dropout_does_not_restart_the_wait(self):
        """The lock survives a frame or two of losing the face; what it must not
        do is start a turn, which is now attention's call."""
        self.runtime._submit=Mock()
        owner={'resolved_person_id':'owner','role':'owner','identity_confidence':.95,'active':True}
        for stamp,seen in [(100000,True),(104000,False),(104200,True),(109500,True)]:
            with patch('brain.runtime.now_ms',return_value=stamp):
                self.runtime.step({'person_hypotheses':[owner] if seen else [],
                                   'attention':{'person_id':'owner','listen':True,'confidence':.9}})
        self.runtime._submit.assert_not_called()
        self.assertEqual(self.runtime.state['initiative']['owner']['wait_remaining_ms'],0)


    def test_cough_care_waits_for_quiet_focus_and_does_not_interrupt(self):
        now=int(time.time()*1000)
        event={'id':'cough1','kind':'cough','stamp_ms':now}
        self.runtime.step({'voice':{'audio_scene':{'events':[event]}}})
        self.runtime.tts.speak.assert_not_called()
        with patch('brain.runtime.now_ms',return_value=now+1200):
            self.runtime.step({'attention':{'target_kind':'person','person_id':'owner','listen':True}})
        self.wait()
        self.assertEqual(self.runtime.tts.speak.call_args.args[0],'慢慢来，别着急。')
        self.runtime.tts.interrupt.assert_not_called()

    def test_group_question_waits_and_another_human_can_take_floor(self):
        stamp=int(time.time()*1000)
        turn={'utterance_id':'group-q','person_id':'a','stamp_ms':stamp,'addressee':'group','transcript':{'text':'大家怎么看','is_final':True,'emitted_ms':stamp},'attention':{'listen':True,'addressed_to_robot':True,'confidence':.8}}
        self.runtime.step({'utterances':[turn]})
        self.runtime.tts.speak.assert_not_called();self.assertIsNotNone(self.runtime.pending_group_turn)
        answer={'utterance_id':'human-answer','person_id':'b','stamp_ms':stamp+100,'transcript':{'text':'我来解释','is_final':True,'emitted_ms':stamp+100},'attention':{'listen':False,'addressed_to_robot':False}}
        with patch('brain.runtime.now_ms',return_value=stamp+100):self.runtime.step({'utterances':[answer]})
        self.assertIsNone(self.runtime.pending_group_turn);self.runtime.tts.speak.assert_not_called()
