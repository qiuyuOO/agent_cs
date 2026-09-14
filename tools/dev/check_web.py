"""Web 接口自检 —— 起一个真服务, 用真 HTTP 打一遍全部端点.

为什么不用 Starlette 的 TestClient: 那需要 httpx 的 ASGI transport (有, 但
本自检要验证的恰恰是"真 uvicorn 起得来 + SSE 真能流"), 所以直接起进程。
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def get(path: str) -> tuple[int, object, dict]:
    req = urllib.request.Request(BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            ctype = r.headers.get("content-type", "")
            if "json" in ctype:
                return r.status, json.loads(raw.decode("utf-8")), dict(r.headers)
            return r.status, raw, dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw.decode("utf-8")), dict(e.headers)
        except Exception:
            return e.code, raw, dict(e.headers)


def post(path: str, body: dict | None = None) -> tuple[int, object]:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw.decode("utf-8"))
        except Exception:
            return e.code, raw


def wait_port(port: int, timeout: float = 40.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.4)
    return False


def main() -> int:
    proc = subprocess.Popen(
        [str(PY), "-m", "cs2clipper.webapp", "--port", str(PORT), "--no-browser"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        if not wait_port(PORT):
            print("服务没起来, 输出:")
            print(proc.stdout.read() if proc.stdout else "")
            return 1
        print(f"服务已启动 {BASE}\n")

        st, body, hdrs = get("/")
        check("首页可访问", st == 200 and b"CS2" in (body if isinstance(body, bytes) else b""),
              f"HTTP {st}")

        for asset in ("/static/app.css", "/static/app.js"):
            st, body, _ = get(asset)
            check(f"静态资源 {asset}", st == 200 and len(body) > 500, f"{len(body)} bytes")

        st, body, _ = get("/api/health")
        check("GET /api/health", st == 200 and body.get("ok"), f"ffmpeg={body.get('ffmpeg')}")

        st, meta, _ = get("/api/meta")
        check("GET /api/meta", st == 200 and len(meta.get("pref_specs", {})) >= 15,
              f"{len(meta.get('pref_specs', {}))} 项偏好")
        check("meta 带画幅预设", bool(meta.get("aspects")), str(meta.get("aspects")))

        st, mus, _ = get("/api/music?dir=" + urllib.parse.quote(str(Path.home())))
        check("GET /api/music (家目录)", st == 200 and "files" in mus,
              f"{len(mus.get('files', []))} 个文件 / {len(mus.get('dirs', []))} 个子目录")
        st, mus2, _ = get("/api/music?dir=" + urllib.parse.quote(r"D:\CloudMusic"))
        check("GET /api/music (真实音乐库)", st == 200 and len(mus2.get("files", [])) >= 3,
              f"{len(mus2.get('files', []))} 个文件")

        st, demos, _ = get("/api/demos")
        check("GET /api/demos", st == 200 and len(demos.get("files", [])) >= 1,
              f"找到 {len(demos.get('files', []))} 个 demo")

        st, bad, _ = get("/api/artifact?path=" + urllib.parse.quote("../../../.env"))
        check("产物路径穿越被拒", st == 403, f"HTTP {st}")

        st, bad, _ = get("/api/artifact?path=" + urllib.parse.quote("cs2clipper/music.py"))
        check("工作区内文件可读", st == 200, f"HTTP {st}")

        st, edl, _ = get("/api/artifact?path=" + urllib.parse.quote("out/review_final3/edl.json"))
        check("读取 EDL 产物", st == 200 and "clips" in edl, f"{len(edl.get('clips', []))} 段")

        st, out, _ = get("/api/outputs?dir=" + urllib.parse.quote("out/review_final3"))
        check("GET /api/outputs", st == 200 and out.get("clips", 0) > 0,
              f"{out.get('clips')} 个片段")

        st, err = post("/api/validate", {"music_path": "no/such.mp3"})
        check("validate 对坏路径报错", st == 400 and not err.get("ok"), str(err.get("error"))[:60])

        st, ok = post("/api/validate", {
            "music_path": r"D:\CloudMusic\在虚无中永存 - 英雄主义.mp3",
            "demo_path": str(meta.get("default_demo", "")),
        })
        check("validate 接受真实素材", st == 200 and ok.get("ok"), str(ok.get("demo", ""))[-40:])

        st, err = post("/api/validate", {"music_path": str(ROOT / "README.md")})
        check("validate 拒绝错误扩展名", st == 400, str(err.get("error"))[:50])

        st, prefs, _ = get("/api/prefs")
        check("GET /api/prefs", st == 200 and "effective" in prefs,
              f"{len(prefs.get('effective', {}))} 项")

        st, prof, _ = get("/api/profile")
        check("GET /api/profile", st == 200 and "profile" in prof,
              f"{len(prof.get('profile', []))} 维画像 / {len(prof.get('observations', []))} 条证据")

        st, runs, _ = get("/api/runs?limit=5")
        check("GET /api/runs", st == 200 and len(runs.get("runs", [])) >= 1,
              f"{runs.get('stats', {}).get('runs')} 次出片")
        run_id = runs["runs"][0]["id"]
        st, clips, _ = get(f"/api/runs/{run_id}/clips")
        check("GET /api/runs/<id>/clips", st == 200 and "clips" in clips,
              f"#{run_id} {len(clips.get('clips', []))} 段")

        # --- 任务: 用最小参数跑一次短出片, 并验证 SSE 真能推事件 ---
        st, job = post("/api/jobs", {
            "music_path": str(ROOT / "work" / "selftest_track.wav"),
            "demo_path": str(meta.get("default_demo", "")),
            "aspect": "square", "max_clips": 2, "max_cards": 4,
            "use_llm": False, "use_prefs": False, "record": False,
            "out_dir": "out/_webtest",
        })
        check("POST /api/jobs 建任务", st == 201 and job.get("ok"), f"job={job.get('job', {}).get('id')}")
        jid = job["job"]["id"]

        kinds, percents, last_seq = [], [], 0
        req = urllib.request.Request(f"{BASE}/api/jobs/{jid}/events")
        t0 = time.time()
        stream_err = None
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                while time.time() - t0 < 300:
                    try:
                        line = r.readline()
                    except (ConnectionAbortedError, ConnectionResetError) as e:
                        stream_err = f"{type(e).__name__}: {e}"
                        break
                    if not line:
                        break
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    d = json.loads(line[5:].strip())
                    if d.get("kind") == "__close__":
                        break
                    kinds.append(d["kind"])
                    if d.get("affects_percent"):
                        percents.append(d.get("percent", 0))
                    last_seq = max(last_seq, d.get("seq", 0))
        except Exception as e:
            stream_err = f"{type(e).__name__}: {e}"
        check("SSE 流正常结束 (无连接中断)", stream_err is None, stream_err or "clean close")
        check("SSE 收到事件流", len(kinds) >= 6, f"{len(kinds)} 条: {sorted(set(kinds))}")
        check("SSE 百分比单调不减", percents == sorted(percents),
              f"{percents[:3]} … {percents[-3:]}")
        check("SSE 最终到 100%", percents and percents[-1] == 100.0, f"{percents[-1] if percents else '-'}")
        check("SSE 有 done 事件", "done" in kinds)

        st, snap, _ = get(f"/api/jobs/{jid}")
        check("任务状态为 done", st == 200 and snap["job"]["status"] == "done",
              f"status={snap['job']['status']} err={snap['job'].get('error')}")
        vp = snap["job"].get("video_path") or ""
        check("任务产出成片", bool(vp) and Path(vp).is_file(),
              f"{Path(vp).name} {Path(vp).stat().st_size // 1024 if vp and Path(vp).is_file() else 0} KB")

        st, _, hdrs = get("/api/download?path=" + urllib.parse.quote(
            str(Path(vp).relative_to(ROOT)) if vp else ""))
        check("成片可下载", st == 200 and hdrs.get("content-type") == "video/mp4", f"HTTP {st}")

        st, ev2, _ = get(f"/api/jobs/{jid}")
        check("已结束任务仍可查询", st == 200, f"HTTP {st}")

        st, canc = post(f"/api/jobs/{jid}/cancel")
        check("对已结束任务取消是幂等空操作", st == 200 and canc.get("note"),
              str(canc.get("note")))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\n" + "=" * 60)
    print(f"总计 {len(PASS) + len(FAIL)} 项: {len(PASS)} 通过, {len(FAIL)} 失败")
    if FAIL:
        print("失败项: " + ", ".join(FAIL))
    print("=" * 60)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
