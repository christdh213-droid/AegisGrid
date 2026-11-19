# SIH26053 LiDAR viewer

Run the synthetic LiDAR demo with Python 3.13. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
$env:PORT = '8766'
$env:MAX_FPS = '12'
$env:DISABLE_PROCESS_POOL = '1'
.\.venv\Scripts\python.exe prototype_app.py --host 0.0.0.0
```

Open [the local viewer](http://127.0.0.1:8766/) and click **Play**. A green **LIVE** indicator means the WebSocket is connected. Verify that the frame number advances. The [health endpoint](http://127.0.0.1:8766/healthz) returns HTTP 200 and `{"status":"ok"}`. For local access only, pass `--host 127.0.0.1` instead.

**Upload LiDAR** previews .csv files with x,y,z columns and little-endian float32 XYZI .bin files up to 8 MiB. The browser displays a fixed-grid elevation view of the file. The live telemetry dock continues to describe the simulation. Use **Return to live** to restore its map.

`PORT` defaults to 8000 locally. `MAX_FPS` caps automatic playback even at 2× speed. It sets a scheduling cap, and actual delivery depends on compute and network load. Set `DISABLE_PROCESS_POOL=0` to enable the optional separate-process fixed-grid route benchmark. The adaptive route works with either setting.

## Deploy on Render

1. Create an empty repository if you don't already have a remote. On GitHub, choose **+ → New repository**, name it, select **Private** if the source should stay private, leave initialization options unchecked, and click **Create repository**. Copy the HTTPS clone URL. In the local checkout, run `git remote add origin <copied-URL>` if `origin` is absent, then `git push -u origin phase2-spatial-ui`. The Render Web Service URL itself is public.
2. In the [Render Dashboard](https://dashboard.render.com/), choose **New → Web Service → Git Provider**. Connect your Git account if prompted, select the repository, and click **Connect**.
3. Enter a service name, select branch **phase2-spatial-ui**, and choose **Docker** as the runtime. Leave **Root Directory** blank if the Dockerfile is at the repository root. For a parent repository, enter the relative directory containing this README and Dockerfile. Use **./Dockerfile** if prompted for the Dockerfile path. Leave **Docker Command** blank to use the image startup command.
4. Select an instance plan and keep one instance for the demo, since playback state is in memory and shared by viewers. Under **Advanced**, set **Health Check Path** to **/healthz**. Add environment variables **MAX_FPS=12** and **DISABLE_PROCESS_POOL=1**. Render supplies **PORT**. Keep auto-deploy off if only approved commits should go live.
5. Click **Create Web Service**. Wait for the deployment to go live, then open the service's HTTPS URL. Check **/healthz**, click **Play** on the main page, and look for green **LIVE** status with advancing frames. The browser derives **wss://** from the HTTPS page URL. For subsequent approved changes, push the branch and choose **Manual Deploy → Deploy latest commit**.

See [Render Web Services](https://render.com/docs/web-services) and [Health Checks](https://render.com/docs/health-checks).

With Docker installed, the equivalent local commands are:

```shell
docker build -t sih26053-viewer .
docker run --rm -p 127.0.0.1:8766:8766 -e PORT=8766 -e MAX_FPS=12 -e DISABLE_PROCESS_POOL=1 sih26053-viewer
```


### Static scan upload
Select **Upload LiDAR** or drop a `.bin` (little-endian float32 XYZI), `.npy`
(real numeric Nx3/Nx4, no pickle), or `.csv` (`x,y,z[,intensity]`, optional header).
Uploads are capped at 50 MiB and 1,000,000 points, processed in memory, cropped to
[-32,+32] m, and normalized by subtracting the in-bounds 10th-percentile Z.
SOR, traversability and goal planning work on the static scan; tracking, dust,
tilt simulation and playback are inactive. **Return to Live Sim** restores the
previous simulation settings and starts paused. The demo has one shared server
state, so an upload changes the scan for every connected viewer.
API: POST raw file bytes to `/api/upload?filename=scan.npy`; return with
`{"action":"return_live"}` on the existing control channel.
