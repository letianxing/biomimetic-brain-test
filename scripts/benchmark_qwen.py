import subprocess,time,json,urllib.request, pathlib
root=pathlib.Path(__file__).resolve().parents[2]/'robo-model'
base=[str(root/'third_party/llama.cpp/build/bin/llama-server'),'-m',str(root/'models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf'),'--host','127.0.0.1','--port','18051','-c','4096','-ngl','99','-np','1','--flash-attn','on','-t','8','-tb','8','-lv','4']
def request(path,payload=None):
 data=json.dumps(payload).encode() if payload is not None else None
 return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request('http://127.0.0.1:18051'+path,data,{'Content-Type':'application/json'}),timeout=90)
results=[]
for batch,ub in [(512,256),(2048,512),(2048,1024)]:
 log=open(f'/tmp/qwen-bench-{ub}.log','w');process=subprocess.Popen(base+['-b',str(batch),'-ub',str(ub)],stdout=log,stderr=log)
 try:
  for i in range(100):
   try:
    with request('/health') as r:json.load(r)
    break
   except Exception:time.sleep(.2)
  prompt='用户说周六下午三点在公园北门见，小明带柠檬水，小天带饼干。'*45+'请回答约定几点见面，只答四个字。'
  for run in range(2):
   start=time.perf_counter();first=None;timing=None;text=''
   with request('/completion',{'prompt':prompt,'n_predict':12,'stream':True,'cache_prompt':False,'temperature':0}) as response:
    for line in response:
     if not line.startswith(b'data: '):continue
     event=json.loads(line[6:])
     if event.get('content'):
      if first is None:first=(time.perf_counter()-start)*1000
      text+=event['content']
     if 'timings'in event:timing=event['timings']
   row={'batch':batch,'ubatch':ub,'run':run,'ttft_ms':round(first or 0,1),'total_ms':round((time.perf_counter()-start)*1000,1),'timings':timing}
   results.append(row);print(json.dumps(row),flush=True)
 finally:
  process.terminate()
  try:process.wait(5)
  except subprocess.TimeoutExpired:process.kill();process.wait()
  log.close()
pathlib.Path('/tmp/qwen-benchmark.json').write_text(json.dumps(results,indent=2))
