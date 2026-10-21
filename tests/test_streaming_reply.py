import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock
from brain.runtime import BrainRuntime
from brain.llm import OpenAICompatibleClient
from brain.streaming_speech import StreamingSpeech


class StreamTests(unittest.TestCase):
    def test_the_reply_is_spoken_once_and_whole(self):
        played = threading.Event()
        producer_observed_playback = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if not payload['stream']:
                    # The default path asks for the whole answer, like the real
                    # server, and nothing is spoken until it arrives.
                    producer_observed_playback.append(played.wait(.3))
                    body = json.dumps({'choices':[{'message':{'content':'第一句。第二句。'}}]},
                                      ensure_ascii=False).encode()
                    self.send_response(200); self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
                    return
                self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
                def send(text):
                    self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':text}}]},ensure_ascii=False)+'\n\n').encode());self.wfile.flush()
                send('第一句。')
                producer_observed_playback.append(played.wait(.3))
                send('第二句。')
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        runtime=BrainRuntime(memory_path=':memory:')
        runtime.llm=OpenAICompatibleClient(f'http://127.0.0.1:{server.server_port}','test')
        runtime.memory.search=Mock(return_value=[])
        runtime.tts=Mock(enabled=True)
        def speak(text, **kwargs):
            kwargs['on_playback'](True);played.set();kwargs['on_playback'](False);return True
        runtime.tts.speak.side_effect=speak
        stamp=int(time.time()*1000)
        try:
            runtime.step({'utterances':[{'utterance_id':'stream-one','stamp_ms':stamp,'person_id':'person1',
                                        'transcript':{'text':'讲两句话','is_final':True,'emitted_ms':stamp,'ended_ms':stamp-100},
                                        'attention':{'listen':True,'addressed_to_robot':True,'confidence':.9}}]})
            deadline=time.monotonic()+4
            while runtime.jobs and time.monotonic()<deadline: time.sleep(.01)
            self.assertFalse(runtime.jobs)
            # Nothing is spoken while the model is still producing: a six-word
            # reply gains nothing from being streamed, and speaking early is how
            # a superseded or retried turn ended up being said twice.
            self.assertEqual(producer_observed_playback,[False])
            self.assertEqual([call.args[0] for call in runtime.tts.speak.call_args_list],['第一句。第二句。'])
            current=runtime.state['current_turn']
            self.assertGreaterEqual(current['playback_started_ms'], current['decision_ready_ms'])
            self.assertEqual(current['status'],'completed')
            kinds=[event['kind'] for event in runtime.state['timeline']]
            self.assertIn('first_token',kinds)
            traces=[json.loads(r[0]) for r in runtime.memory.db.execute('SELECT payload FROM outbox')]
            self.assertTrue(any(r['kind']=='brain_trace' and r['entity']['track_id']=='stream-one' for r in traces))
        finally:
            runtime.stop();server.shutdown();server.server_close()

    def test_interrupt_discards_remaining_phrases(self):
        cancel=threading.Event();began=threading.Event()
        tts=Mock()
        def speak(text,**kwargs): began.set();cancel.wait(2);return False
        tts.speak.side_effect=speak
        stream=StreamingSpeech(tts,cancel,lambda text:None,lambda text,active:None)
        stream.feed('第一句。第二句。')
        self.assertTrue(began.wait(1))
        cancel.set();stream.finish()
        self.assertEqual(tts.speak.call_count,1)

    def test_social_yield_waits_for_current_chunk_and_discards_next(self):
        cancel=threading.Event();began=threading.Event();release=threading.Event()
        tts=Mock()
        def speak(text,**kwargs):began.set();release.wait(1);return True
        tts.speak.side_effect=speak
        stream=StreamingSpeech(tts,cancel,lambda text:None,lambda text,active:None)
        stream.feed('第一句。第二句。');self.assertTrue(began.wait(1))
        yielded=threading.Event()
        def finish():cancel.set();yielded.set()
        stream.request_yield(finish)
        self.assertFalse(yielded.is_set());release.set();stream.finish()
        self.assertTrue(yielded.is_set());tts.speak.assert_called_once()
