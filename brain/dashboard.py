from __future__ import annotations

import argparse, json, signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from .runtime import BrainRuntime

HTML='''<!doctype html><meta charset="utf-8"><title>Biomimetic Brain</title>
<style>body{font:16px system-ui;margin:30px;max-width:1000px}section{border:1px solid #ccd8d3;border-radius:12px;padding:18px;margin:14px 0}label{margin-right:18px}pre{white-space:pre-wrap}button,select{font:inherit;padding:7px}</style>
<nav><a href="http://127.0.0.1:8092">打开实时感知总览：摄像头 / 情绪 / 手势 / ASR / VAD / TTS</a></nav><h1>仿生级决策测试大脑</h1><section><b>资格门 → 快脑/慢脑 → 仲裁 → TTS → 动作意图</b></section>
<section><label><input id="enabled" type="checkbox" checked>TTS</label><label>声音<select id="voice"><option>Tingting</option><option>Meijia</option><option>Sinji</option></select></label><button onclick="save()">应用</button></section>
<section><h3>听到的对话与记忆状态</h3><pre id="heard">--</pre><pre id="memory">--</pre></section><section><h3>声学、音乐与反射意图</h3><pre id="audioScene">--</pre><pre id="audioActions">--</pre></section><section><h3>语音回话状态</h3><pre id="s2s">--</pre></section><section><h3>当前决策</h3><pre id="decision">--</pre></section><section><h3>事件</h3><pre id="events">--</pre></section>
<script>async function poll(){let s=await(await fetch('/api/state')).json();audioScene.textContent=JSON.stringify(s.audio_scene,null,2);audioActions.textContent=JSON.stringify(s.audio_actions,null,2);heard.textContent=JSON.stringify(s.heard,null,2);memory.textContent=JSON.stringify(s.memory,null,2);s2s.textContent=JSON.stringify(s.s2s,null,2);decision.textContent=JSON.stringify(s.decision,null,2);events.textContent=JSON.stringify(s.events,null,2)}async function save(){let q=new URLSearchParams({enabled:enabled.checked,voice:voice.value});await fetch('/api/tts?'+q,{method:'POST'});poll()}setInterval(poll,500);poll()</script>'''

def serve(host='127.0.0.1', port=8094):
    runtime=BrainRuntime(); runtime.start()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/': return self.send(HTML,'text/html; charset=utf-8')
            if urlparse(self.path).path=='/api/state': return self.send(json.dumps(runtime.snapshot(compact=parse_qs(urlparse(self.path).query).get('view')==['dashboard']),ensure_ascii=False),'application/json')
            if self.path=='/api/health': return self.send(json.dumps({"ok": True, "running": runtime.running},ensure_ascii=False),'application/json')
            self.send_error(404)
        def do_POST(self):
            p=urlparse(self.path)
            if p.path=='/api/tts':
                q=parse_qs(p.query); runtime.configure_tts(q.get('enabled',['true'])[0]=='true',q.get('voice',['Tingting'])[0],q.get('rate',[None])[0]); return self.send('{}','application/json')
            if p.path=='/api/forget':
                q=parse_qs(p.query)
                result=runtime.forget_person(q.get('person',[''])[0],
                                             q.get('all_sessions',['true'])[0]!='false')
                return self.send(json.dumps(result,ensure_ascii=False),'application/json')
            if p.path=='/api/llm':
                # 切换慢脑。云端后端意味着对话内容离开本机，所以这是显式动作。
                q=parse_qs(p.query)
                result=runtime.set_llm(q.get('backend',[''])[0])
                return self.send(json.dumps(result,ensure_ascii=False),'application/json')
            self.send_error(404)
        def send(self,text,ctype):
            data=text.encode(); self.send_response(200); self.send_header('content-type',ctype); self.send_header('content-length',len(data)); self.end_headers(); self.wfile.write(data)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer((host,port),Handler)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: runtime.stop(); server.server_close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8094);a=p.parse_args();serve(a.host,a.port)
if __name__=='__main__': main()
