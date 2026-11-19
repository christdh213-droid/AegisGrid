"""Session 6 browser acceptance on an isolated localhost server and Chrome profile."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]

def main():
    chrome = Path(os.environ.get('PROGRAMFILES', r'C:\Program Files')) / 'Google/Chrome/Application/chrome.exe'
    log = open(ROOT/'output/session6_server.log', 'w')
    env = dict(os.environ, MAX_FPS='12', DISABLE_PROCESS_POOL='1')
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'prototype_app:app', '--host', '127.0.0.1', '--port', '8766'], cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    browser = None
    try:
        for _ in range(100):
            try:
                assert urllib.request.urlopen('http://127.0.0.1:8766/healthz', timeout=1).status == 200
                break
            except OSError: time.sleep(.1)
        with tempfile.TemporaryDirectory(prefix='sih-session6-') as profile:
            browser = subprocess.Popen([str(chrome), '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--hide-scrollbars', f'--user-data-dir={profile}', '--remote-debugging-port=0', '--window-size=1600,1100', 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            for _ in range(100):
                portfile = Path(profile)/'DevToolsActivePort'
                if portfile.exists(): break
                time.sleep(.1)
            port = portfile.read_text().splitlines()[0]
            tab = next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json')) if t['type']=='page')
            events=[]; counter=0; checks={}
            with connect(tab['webSocketDebuggerUrl'], max_size=20_000_000) as ws:
                def call(method, params=None):
                    nonlocal counter
                    counter+=1; ident=counter
                    ws.send(json.dumps(dict(id=ident,method=method,params=params or {})))
                    while True:
                        msg=json.loads(ws.recv(timeout=30))
                        if msg.get('id')==ident:
                            if 'error' in msg: raise RuntimeError(msg['error'])
                            return msg.get('result',{})
                        events.append(msg)
                def evaluate(expression):
                    result=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
                    if 'exceptionDetails' in result: raise RuntimeError(result['exceptionDetails'])
                    return result.get('result',{}).get('value')
                def check(name, expression):
                    for _ in range(40):
                        value=evaluate(expression)
                        if value: checks[name]=True; return
                        time.sleep(.1)
                    raise AssertionError(name)
                def click(selector): evaluate(f'document.querySelector({json.dumps(selector)}).click()')
                def value(selector, val): evaluate(f"(()=>{{const e=document.querySelector({json.dumps(selector)});e.value={json.dumps(str(val))};e.dispatchEvent(new Event('input'));e.dispatchEvent(new Event('change'));}})()")
                def config(key, expected): check(key, f"fetch('/api/frame').then(r=>r.json()).then(f=>f.extras.runtime_config[{json.dumps(key)}]==={json.dumps(expected)})")
                call('Runtime.enable');call('Network.enable');call('Page.enable');call('DOM.enable')
                call('Page.navigate',{'url':'http://127.0.0.1:8766/'})
                check('connected', "document.querySelector('#connection')?.hasAttribute('data-live')")
                click('#play');time.sleep(5)
                diagnostics=evaluate('window.viewerDiagnostics')
                assert diagnostics['framesReceived']>10 and diagnostics['renderFps']>0
                click('#pause');time.sleep(.3)
                click('[data-config=sor_enabled]'); config('sor_enabled',False)
                click('[data-config=sor_enabled]');config('sor_enabled',True)
                value('#dust',15);config('outlier_ratio',.15)
                click('[data-config=simulate_hill_tilt]');config('simulate_hill_tilt',True)
                value('#tiltMode','Manual');config('tilt_mode','Manual')
                value('#pitch',5);config('pitch_deg',5)
                value('#roll',-3);config('roll_deg',-3)
                click('[data-config=imu_levelling_enabled]');config('imu_levelling_enabled',False)
                check('uncompensated',"!document.querySelector('#uncompensated').hidden")
                click('[data-config=imu_levelling_enabled]');config('imu_levelling_enabled',True)
                click('[data-config=planner_enabled]');config('planner_enabled',False)
                click('[data-config=planner_enabled]');config('planner_enabled',True)
                for mode in ['elevation','quadtree','risk','split']:
                    click(f'[data-mode={mode}]');check('view_'+mode,f"document.querySelector('[data-mode={mode}]').getAttribute('aria-pressed')==='true'")
                value('#governorProfile','CPU Load Spike Stress Test');config('governor_profile','CPU Load Spike Stress Test')
                click('#play');time.sleep(2);click('#pause')
                check('governor',"document.querySelector('#governorReadout').textContent.includes('Emulated')")
                click('#hideHud');check('hidden_hud',"document.querySelector('#telemetry').hidden")
                click('#hideHud');check('shown_hud',"!document.querySelector('#telemetry').hidden")
                def mouse(kind,x,y,**kw): call('Input.dispatchMouseEvent',dict(type=kind,x=x,y=y,**kw))
                handle=evaluate("(()=>{const r=document.querySelector('#wipe').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()")
                mouse('mousePressed',**handle,button='left',clickCount=1);mouse('mouseMoved',handle['x']+100,handle['y'],button='left',buttons=1);mouse('mouseReleased',handle['x']+100,handle['y'],button='left',clickCount=1)
                check('wipe',"parseFloat(document.querySelector('#wipe').style.left)>50")
                mouse('mouseMoved',600,500);check('tooltip',"!document.querySelector('#tooltip').hidden")
                mouse('mousePressed',500,450,button='left',clickCount=1);mouse('mouseReleased',500,450,button='left',clickCount=1)
                check('goal_click',"fetch('/api/frame').then(r=>r.json()).then(f=>f.extras.runtime_config.goal_xy!==null)")
                before=evaluate("document.querySelector('#rightCanvas').toDataURL()")
                mouse('mousePressed',400,400,button='left',clickCount=1);mouse('mouseMoved',450,420,button='left',buttons=1);mouse('mouseReleased',450,420,button='left',clickCount=1);time.sleep(.2)
                checks['pan']=before!=evaluate("document.querySelector('#rightCanvas').toDataURL()")
                mouse('mouseWheel',500,450,deltaX=0,deltaY=-200);time.sleep(.2)
                check('aria_switches', "[...document.querySelectorAll('[role=switch]')].every(e=>e.getAttribute('aria-checked')===String(e.checked))")
                evaluate("document.querySelector('[data-mode=split]').focus()")
                call('Input.dispatchKeyEvent',{'type':'keyDown','key':'ArrowRight','code':'ArrowRight'})
                call('Input.dispatchKeyEvent',{'type':'keyUp','key':'ArrowRight','code':'ArrowRight'})
                check('keyboard_modes',"document.querySelector('[data-mode=elevation]').getAttribute('aria-pressed')==='true'")
                evaluate("document.querySelector('#rightCanvas').focus()")
                call('Input.dispatchKeyEvent',{'type':'keyDown','key':'h','code':'KeyH'})
                call('Input.dispatchKeyEvent',{'type':'keyUp','key':'h','code':'KeyH'})
                check('keyboard_hud',"document.querySelector('#telemetry').hidden")
                click('#hideHud');click('[data-mode=split]')
                before=evaluate("document.querySelector('#rightCanvas').toDataURL()")
                call('Emulation.setTouchEmulationEnabled',{'enabled':True,'maxTouchPoints':2})
                call('Input.dispatchTouchEvent',{'type':'touchStart','touchPoints':[{'x':450,'y':450,'id':1},{'x':550,'y':450,'id':2}]})
                call('Input.dispatchTouchEvent',{'type':'touchMove','touchPoints':[{'x':400,'y':450,'id':1},{'x':600,'y':450,'id':2}]})
                call('Input.dispatchTouchEvent',{'type':'touchEnd','touchPoints':[]})
                time.sleep(.2)
                checks['pinch']=before!=evaluate("document.querySelector('#rightCanvas').toDataURL()")
                call('Emulation.setTouchEmulationEnabled',{'enabled':False})
                click('#home');click('#reset');time.sleep(.3)
                scan=ROOT/'output/session6_scan.csv';scan.write_text('x,y,z\n'+'\n'.join(f'{x/4},{y/4},-1.7' for x in range(-20,21) for y in range(-20,21)))
                root=call('DOM.getDocument')['root']['nodeId'];node=call('DOM.querySelector',{'nodeId':root,'selector':'#file'})['nodeId']
                call('DOM.setFileInputFiles',{'nodeId':node,'files':[str(scan)]})
                check('upload',"!document.querySelector('#returnLive').hidden")
                check('static_live_controls_disabled',"document.querySelector('#play').disabled && document.querySelector('#dust').disabled")
                click('[data-config=sor_enabled]');config('sor_enabled',False)
                value('#goalX',2);value('#goalY',2);evaluate("document.querySelector('#goalForm').requestSubmit()")
                check('static_planner',"document.querySelector('#routeStatus').textContent.includes('OPTIMAL')")
                click('#returnLive');check('return_live',"document.querySelector('#returnLive').hidden && !document.querySelector('#play').disabled")
                scan.write_text('invalid')
                call('DOM.setFileInputFiles',{'nodeId':node,'files':[str(scan)]})
                check('error_toast',"document.querySelector('#error').textContent.length>0")
                evaluate("document.querySelector('#error').textContent=''")
                click('#reset');click('#play');time.sleep(2)
                shot=call('Page.captureScreenshot',{'format':'png'});(ROOT/'output/session6_viewer.png').write_bytes(base64.b64decode(shot['data']))
                click('#pause')
                hosts=sorted({e['params']['request']['url'].split('/')[2] for e in events if e.get('method')=='Network.requestWillBeSent' and '://' in e['params']['request']['url']})
                errors=[e['params'] for e in events if e.get('method')=='Runtime.exceptionThrown']
                assert not errors,errors
                assert hosts==['127.0.0.1:8766'],hosts
                assert all(checks.values()),checks
                result=dict(checks=checks,diagnostics=diagnostics,request_hosts=hosts,exceptions=errors,health_http_status=200)
                (ROOT/'output/session6_browser.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
            browser.terminate();browser.wait(10);browser=None
    finally:
        if browser is not None:browser.terminate();browser.wait(10)
        server.terminate();server.wait(10);log.close()

if __name__=='__main__':main()
