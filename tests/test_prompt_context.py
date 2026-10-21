import os,unittest,unittest.mock,json
from brain.prompt_context import build_context,context_budget,estimate_tokens,fit_to_budget
class ContextSizeTests(unittest.TestCase):
    """The server refuses an over-long request outright, so the turn is lost."""
    def long_context(self):
        line='我们刚才讨论的是下午三点那个会议要不要提前，另外他还提到上次那个方案需要重新核对一遍数据'
        turns=[{'person_id':'天行','text':line,'stamp_ms':1789551300000+i*1000,'to_robot':True,'session_id':'s1',
                'background':{'visible_participants':2,'speech_act':'statement'}} for i in range(16)]
        memories=[{'kind':'heard_utterance','text':line,'observed_at_ms':1789551200000+i*1000,'trace_id':'m%d'%i,
                   'entity':{'person_id':'天行'},'scope':{'session_id':'s1'}} for i in range(12)]
        return memories,{'current_speaker':{'person_id':'天行','session_id':'s1','recent_conversation':turns}}

    def test_oversized_context_is_trimmed_and_says_what_it_dropped(self):
        memories,perception=self.long_context()
        with unittest.mock.patch.dict(os.environ,{'BRAIN_LLM_CONTEXT_TOKENS':'1800'}):
            prompt,context=build_context('今天天气怎么样啊',memories,perception=perception)
            size=estimate_tokens(prompt)+estimate_tokens(json.dumps(context,ensure_ascii=False,separators=(',',':')))
            self.assertLessEqual(size,context_budget())
        self.assertEqual(context['current_user_text'],'今天天气怎么样啊')
        self.assertTrue(context['recent'])
        dropped=context['context_limits']['dropped_for_context_size']
        self.assertGreater(dropped['history']+dropped['recent'],0)

    def test_history_goes_before_the_live_conversation(self):
        context={'history':[{'text':'旧'*50} for _ in range(6)],'recent':[{'text':'新'*50} for _ in range(6)],
                 'current_user_text':'现在这句'}
        _,trimmed=fit_to_budget('',context,300)
        dropped=trimmed['context_limits']['dropped_for_context_size']
        self.assertEqual(trimmed['history'],[])          # 召回历史先让位
        self.assertEqual(dropped['history'],6)
        self.assertLess(dropped['recent'],6)             # 现场对话最后一句永远保留
        self.assertTrue(trimmed['recent'])

    def test_millisecond_stamps_and_session_ids_stay_out_of_the_prompt(self):
        memories,perception=self.long_context()
        _,context=build_context('嗯',memories,perception=perception)
        body=json.dumps(context,ensure_ascii=False)
        self.assertNotIn('1789551300000',body)
        self.assertNotIn('"s1"',body)  # 会话 id 由 earlier_session 标记代替


class ContextTests(unittest.TestCase):
    def test_preserves_facts_conflict_and_deduplicates_recent(self):
        memory={'kind':'heard_utterance','text':'下午三点在北门','observed_at_ms':1000,'entity':{'person_id':'a'},'structured':{'attention':{'addressed_to_robot':False}}}
        p={'current_speaker':{'person_id':'a','attention':{'identity_conflict':True,'reasons':['x']*200},'recent_conversation':[{'person_id':'a','text':'下午三点在北门','stamp_ms':1000,'to_robot':False}]}}
        prompt,context=build_context('你说呢',[memory],perception=p)
        self.assertEqual(context['recent'][0]['text'],'下午三点在北门')
        self.assertEqual(context['history'],[])
        self.assertTrue(context['current_speaker']['identity_conflict'])
        self.assertNotIn('reasons',json.dumps(context))
        self.assertIn('输入已通过对话门控',prompt)
    def test_same_words_from_different_people_and_times_are_preserved(self):
        memories=[{'text':'三点','observed_at_ms':i,'entity':{'person_id':p}} for i,p in [(1,'a'),(2,'b')]]
        _,context=build_context('谁说的',memories)
        self.assertEqual(len(context['history']),2)

    def test_short_gentle_style_applies_to_normal_and_proactive(self):
        for owner in [None,{'role':'owner'}]:
            prompt,_=build_context('今天不开心',[],proactive=owner)
            self.assertIn('不超过 6 个字',prompt)   # 用户要的默认长度
            self.assertIn('语气温柔自然',prompt)
            self.assertIn('事实问题简洁准确',prompt)
            self.assertIn('不要用反问开场',prompt)   # 现场问题：回复几乎都是反问句

    def test_general_followup_preserves_sequence_and_background_without_topic_rule(self):
        perception={'current_speaker':{'person_id':'owner','recent_conversation':[
            {'person_id':'stranger','text':'你叫什么名字啊','stamp_ms':1000,'to_robot':True},
            {'person_id':'stranger','text':'今天天气怎么样呢','stamp_ms':7000,'to_robot':False,
             'identity':{'identity_conflict':True},'background':{'visible_participants':['owner'],'visual_observations':[{'person_id':'owner','gaze_score':.8,'emotion_valid':False,'emotion_label':'sad'}]}}]}}
        for text in ['你说呢','那这事怎么办','这主意靠谱吗']:
            prompt,context=build_context(text,[],perception=perception)
            self.assertNotIn('current_topic',context)
            self.assertEqual([r['t'] for r in context['recent']],[-6,0])  # 秒，相对当前发言
            self.assertFalse(context['recent'][1]['to_robot'])
            self.assertTrue(context['recent'][1]['identity']['identity_conflict'])
            self.assertNotIn('expression',context['recent'][1]['background']['visual_observations'][0])
            self.assertIn('自主判断话题延续',prompt)

    def test_group_context_keeps_multiple_speakers_and_addressees(self):
        turns=[{'person_id':str(i),'text':'方案'+str(i),'stamp_ms':i*1000,'background':{'addressee':'group'}} for i in range(6)]
        perception={'current_speaker':{'person_id':'5','group_context':{'active':True,'participants':['0','1','2','3','4','5','robot']},'recent_conversation':turns}}
        prompt,context=build_context('比较一下大家的意见',[],perception=perception)
        self.assertEqual(len(context['recent']),6)
        self.assertEqual(len(context['group_conversation']['participants']),7)
        self.assertTrue(all(r['background']['addressee']=='group' for r in context['recent']))
        self.assertIn('到两句',prompt)            # 比较多人观点时允许说完
        self.assertIn('不把不同人的观点合并成同一个人',prompt)
