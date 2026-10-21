import time
import unittest
from unittest.mock import Mock, patch
from brain.registration import parse_registration
from brain.runtime import BrainRuntime

class VoiceRegistrationTests(unittest.TestCase):
    def test_fixed_phrase_and_asr_punctuation(self):
        for text in ['你好小圆，我叫小天，请记住我，我是你的主人。','你好，小圆。我叫小天请记住我我是你的主人']:
            self.assertEqual(parse_registration(text),{'name':'小天','role':'owner'})
        self.assertEqual(parse_registration('小圆，我叫person2，请记住我。'),{'name':'person2','role':'known'})
        self.assertIsNone(parse_registration('他说你好小圆我叫小天请记住我'))
        self.assertIsNone(parse_registration('小圆，我叫小天，今天很开心'))

    def test_spoken_request_registers_and_replies_without_llm_or_browser(self):
        runtime=BrainRuntime(memory_path=':memory:');runtime.tts=Mock(enabled=True)
        runtime.tts.speak.return_value=True;runtime.memory.search=Mock(return_value=[]);runtime.llm=Mock()
        self.addCleanup(runtime.stop)
        stamp=int(time.time()*1000)
        turn={'utterance_id':'register1','person_id':'unknown','stamp_ms':stamp,'transcript':{'is_final':True,'emitted_ms':stamp,'text':'你好小圆，我叫小天，请记住我，我是你的主人。'},'attention':{'listen':True,'addressed_to_robot':True,'confidence':.9}}
        with patch('brain.runtime.post_json',return_value={'success':True,'person_id':'小天'}) as post:
            runtime.step({'utterances':[turn]})
            deadline=time.monotonic()+2
            while runtime.jobs and time.monotonic()<deadline:time.sleep(.01)
            self.assertFalse(runtime.jobs)
            post.assert_called_once()
            self.assertEqual(post.call_args.args[1]['utterance_id'],'register1')
        runtime.tts.speak.assert_called_once()
        self.assertIn('你是我的主人了',runtime.tts.speak.call_args.args[0])
        runtime.llm.respond.assert_not_called();runtime.memory.search.assert_not_called()
        runtime.step({'utterances':[turn]})
        runtime.tts.speak.assert_called_once()

    def test_unaddressed_registration_is_only_remembered(self):
        runtime=BrainRuntime(memory_path=':memory:');runtime.tts=Mock();self.addCleanup(runtime.stop)
        stamp=int(time.time()*1000)
        with patch('brain.runtime.post_json') as post:
            runtime.step({'utterances':[{'utterance_id':'no','person_id':'unknown','stamp_ms':stamp,'transcript':{'is_final':True,'emitted_ms':stamp,'text':'你好小圆我叫小天请记住我'},'attention':{'listen':False,'addressed_to_robot':False}}]})
        post.assert_not_called();runtime.tts.speak.assert_not_called()
