"""Session 1 measurements: 300 frames, real ASGI sends, optional Chrome smoke.

No final stress matrix or future modules. No timing assertions that hide host load.
"""
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stream_runtime import StreamState
from src.stream_protocol import decode_frame

ROOT = Path(__file__).resolve().parents[1]

def percentiles(values):
    return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)))

def compute_measurement():
    state = StreamState()
    stages = {k: [] for k in ('generate','pre_process','fixed','track','adaptive','post_map','wall','payload_bytes')}
    def wrap(obj, name, key):
        original = getattr(obj, name)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            stages[key].append((time.perf_counter() - start) * 1000)
            return result
        setattr(obj, name, timed)
    wrap(state.stream,'_generate_frame','generate')
    for obj,name,key in ((state.pipeline,'pre_hook','pre_process'),(state.pipeline.fixed_map,'update','fixed'),(state.pipeline.tracker,'process','track'),(state.pipeline.adaptive_map,'build','adaptive'),(state.pipeline,'post_hook','post_map')):
        wrap(obj,name,key)
    metrics=[]
    for _ in range(300):
        start=time.perf_counter()
        state.execute({'action':'step'})
        stages['wall'].append((time.perf_counter()-start)*1000)
        stages['payload_bytes'].append(len(state.latest_binary))
        metrics.append(state.current_frame.metrics)
    return dict(frames=300,stages={k:percentiles(v) for k,v in stages.items()},pack_ms=percentiles(state.pack_samples),
                unpaced_fps=1000/np.mean(stages['wall']),
                golden=dict(max_rmse_m=max(m.obstacle_rmse_m for m in metrics),min_cell_reduction_pct=min(m.cell_reduction_pct for m in metrics),min_memory_saved_pct=min(m.memory_saved_pct for m in metrics),ghost_cells_total=sum(m.ghost_cells_detected for m in metrics)))

async def transport_measurement():
    from prototype_app import app,state
    async with app.router.lifespan_context(app):
        await asyncio.to_thread(state.execute, {'action':'reset'})
        sent=[]
        command_queue=asyncio.Queue()
        await command_queue.put({'type':'websocket.connect'})
        await command_queue.put({'type':'websocket.receive','text':'{"action":"play"}'})
        started=time.perf_counter()
        async def receive():
            return await command_queue.get()
        async def send(message):
            # Real ASGI endpoint, measuring awaited send; socket transport separately below.
            if message['type']=='websocket.send' and 'bytes' in message:
                sent.append(decode_frame(message['bytes'])['frame_id'])
                if len(sent)>=301:
                    await command_queue.put({'type':'websocket.disconnect','code':1000})
        scope=dict(type='websocket',path='/ws/stream',raw_path=b'/ws/stream',query_string=b'',headers=[],scheme='ws',client=('127.0.0.1',1),server=('127.0.0.1',8000),subprotocols=[])
        await app(scope,receive,send)
        elapsed=time.perf_counter()-started
        samples=list(state.send_samples)
        return dict(frames=len(sent),observed_server_fps=(len(sent)-1)/elapsed,
                    pack_plus_asgi_send_ms=percentiles([s['encode_ms'] for s in samples]),
                    note='ASGI sink send; kernel/network send is measured by the real-server browser/WS probe')

def browser_measurement():
    """Existing Chrome via CDP, isolated headless profile; no packages downloaded."""
    from websockets.sync.client import connect
    chrome=Path(os.environ.get('PROGRAMFILES', r'C:\Program Files'))/'Google/Chrome/Application/chrome.exe'
    if not chrome.exists():
        return {'verified':False,'reason':'Chrome not found'}
    server_log=open(ROOT/'output/session1_server.log','w')
    server=subprocess.Popen([sys.executable,'-m','uvicorn','prototype_app:app','--port','8765','--log-level','warning','--ws-per-message-deflate','false'],cwd=ROOT,stdout=server_log,stderr=server_log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    browser=None
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen('http://127.0.0.1:8765/api/frame',timeout=1).close();break
            except OSError:time.sleep(.1)
        else:raise RuntimeError('Server did not start')
        # Kernel-backed localhost WebSocket, 300 live deliveries after initial frame.
        with connect('ws://127.0.0.1:8765/ws/stream', max_size=10_000_000) as stream:
            stream.recv(timeout=10)
            stream.send('{"action":"reset"}')
            while True:
                message=stream.recv(timeout=10)
                if isinstance(message,str) and json.loads(message).get('type')=='control_ack':break
            stream.send('{"action":"play"}')
            frames=0;start=time.perf_counter()
            while frames<300:
                if isinstance(stream.recv(timeout=20),bytes):frames+=1
            stream.send('{"action":"pause"}')
            socket_fps=frames/(time.perf_counter()-start)
            time.sleep(.1)
            socket_stats=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/stream-stats'))
            socket_stats['observed_receive_fps']=socket_fps
        with tempfile.TemporaryDirectory(prefix='sih-session1-') as profile:
            browser=subprocess.Popen([str(chrome),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',f'--user-data-dir={profile}','--remote-debugging-port=0','--window-size=1440,1000','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            port_file=Path(profile)/'DevToolsActivePort'
            for _ in range(100):
                if port_file.exists():break
                time.sleep(.1)
            port=port_file.read_text().splitlines()[0]
            tabs=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
            tab=next(t for t in tabs if t['type']=='page')
            events=[];counter=0
            with connect(tab['webSocketDebuggerUrl'],max_size=10_000_000) as ws:
                def call(method,params=None):
                    nonlocal counter
                    counter+=1;ident=counter
                    ws.send(json.dumps(dict(id=ident,method=method,params=params or {})))
                    while True:
                        msg=json.loads(ws.recv(timeout=30))
                        if msg.get('id')==ident:
                            if 'error' in msg:raise RuntimeError(msg['error'])
                            return msg.get('result',{})
                        events.append(msg)
                def evaluate(expression):
                    result=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
                    if 'exceptionDetails' in result:raise RuntimeError(result['exceptionDetails'])
                    return result.get('result',{}).get('value')
                call('Runtime.enable');call('Network.enable');call('Page.enable')
                call('Page.navigate',{'url':'http://127.0.0.1:8765/'})
                time.sleep(2)
                evaluate("document.getElementById('play').click()")
                time.sleep(6)
                diagnostics=evaluate('window.viewerDiagnostics')
                shot=call('Page.captureScreenshot',{'format':'png'})
                (ROOT/'output/session1_viewer.png').write_bytes(base64.b64decode(shot['data']))
                evaluate("document.getElementById('pause').click()")
                time.sleep(.2)
                # Exercise real browser pointer events against the handle and canvas.
                handle=evaluate("(()=>{const r=document.getElementById('wipe').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()")
                def drag(x,y,dx,dy):
                    call('Input.dispatchMouseEvent',dict(type='mousePressed',x=x,y=y,button='left',clickCount=1))
                    call('Input.dispatchMouseEvent',dict(type='mouseMoved',x=x+dx,y=y+dy,button='left',buttons=1))
                    call('Input.dispatchMouseEvent',dict(type='mouseReleased',x=x+dx,y=y+dy,button='left',clickCount=1))
                drag(handle['x'],handle['y'],150,0)
                wipe_moved=evaluate("parseFloat(document.getElementById('wipe').style.left)>55")
                initial_pixels=evaluate("document.getElementById('rightCanvas').toDataURL()")
                drag(300,300,70,40);time.sleep(.1)
                pan_changed=initial_pixels!=evaluate("document.getElementById('rightCanvas').toDataURL()")
                before_zoom=evaluate("document.getElementById('rightCanvas').toDataURL()")
                call('Input.dispatchMouseEvent',dict(type='mouseWheel',x=500,y=400,deltaX=0,deltaY=-250))
                time.sleep(.1)
                zoom_changed=before_zoom!=evaluate("document.getElementById('rightCanvas').toDataURL()")
                before=evaluate("document.getElementById('metrics').textContent")
                evaluate("document.getElementById('step').click()")
                time.sleep(1)
                after=evaluate("document.getElementById('metrics').textContent")
                evaluate("document.getElementById('reset').click()")
                time.sleep(1)
                reset=evaluate("document.getElementById('metrics').textContent")
                # Explicitly emulate a browser without WebSocket, then exercise REST.
                call('Page.addScriptToEvaluateOnNewDocument',dict(source="window.WebSocket=class {constructor(){throw new Error('WebSocket disabled for fallback check')}}"))
                call('Page.reload');time.sleep(2)
                fallback_state=evaluate("document.getElementById('connection').textContent")
                fallback_before=evaluate("document.getElementById('metrics').textContent")
                evaluate("document.getElementById('step').click()");time.sleep(1)
                fallback_step=fallback_before!=evaluate("document.getElementById('metrics').textContent")
                # Browser request monitor includes WS and local assets.
                hosts=sorted({e['params']['request']['url'].split('/')[2] for e in events if e.get('method')=='Network.requestWillBeSent' and '://' in e['params']['request']['url']})
                errors=[e['params'] for e in events if e.get('method')=='Runtime.exceptionThrown']
                result=dict(verified=True,diagnostics=diagnostics,request_hosts=hosts,console_exceptions=errors,step_changed=before!=after,reset_frame_zero='Frame 0 ' in reset,localhost_websocket=socket_stats,wipe_moved=wipe_moved,pan_changed=pan_changed,zoom_changed=zoom_changed,rest_fallback=fallback_state,rest_step_changed=fallback_step)
            browser.terminate();browser.wait(timeout=10);browser=None
            return result
    finally:
        if browser is not None:
            browser.terminate();browser.wait(timeout=10)
        server.terminate();server.wait(timeout=10);server_log.close()

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--browser-only',action='store_true')
    args=parser.parse_args()
    if args.browser_only:
        result=browser_measurement()
        (ROOT/'output/session1_browser.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        return
    result={}
    try:
        result['compute']=compute_measurement()
        print('Compute:',json.dumps(result['compute']),flush=True)
        result['transport']=asyncio.run(transport_measurement())
        print('Transport:',json.dumps(result['transport']),flush=True)
        try:result['browser']=browser_measurement()
        except Exception as error:result['browser']={'verified':False,'reason':str(error)}
    finally:
        (ROOT/'output/session1_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
