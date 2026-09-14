import json
import urllib.request

base = "http://127.0.0.1:8760"


def get(path):
    with urllib.request.urlopen(base + path, timeout=40) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def post(path, body=None):
    req = urllib.request.Request(
        base + path, data=json.dumps(body or {}).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


st, h = get("/api/health")
print("health:", {"ok": h["ok"], "ffmpeg": h["ffmpeg"]})

st, s = get("/api/hlae/status")
pf = s["preflight"]
print(f"\n/api/hlae/status  HTTP {st}")
print(f"  就绪={pf['ok']}  问题={len(pf['problems'])}  提醒={len(pf['warnings'])}")
print(f"  HLAE={pf['info'].get('hlae_version')}  cs2={pf['info'].get('cs2_size_gb')} GB")
print(f"  录制目录={s['record_dir']}")
print(f"  已有素材={s['has_material']}")
print(f"  采集帧率={s['fps']}")

st, m = get("/api/meta")
print(f"\n/api/meta  record_source={m.get('record_source')}  偏好项={len(m['pref_specs'])}")

# 建任务: 走 hlae 源, 应当先报"还没有录制素材"并给出下一步
st, job = post("/api/jobs", {
    "music_path": r"E:\agent_cs\work\selftest_track.wav",
    "aspect": "square", "max_clips": 2, "max_cards": 4,
    "use_llm": False, "use_prefs": False, "record": False,
    "record_source": "hlae", "out_dir": "out/_hlae_webtest",
})
print(f"\nPOST /api/jobs (record_source=hlae)  HTTP {st}  job={job['job']['id']}")

req = urllib.request.Request(base + f"/api/jobs/{job['job']['id']}/events")
msgs = []
with urllib.request.urlopen(req, timeout=300) as r:
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        d = json.loads(line[5:].strip())
        if d.get("kind") == "__close__":
            break
        msgs.append(d)

err = [m for m in msgs if m["kind"] == "error"]
print(f"  事件 {len(msgs)} 条, 其中 error {len(err)} 条")
if err:
    text = err[-1]["message"]
    print("  报错内容 (前 360 字):")
    for line in text.splitlines()[:14]:
        print("   |", line)
    detail = err[-1].get("detail") or {}
    steps = (detail.get("next_steps") or []) if isinstance(detail, dict) else []
    if steps:
        print("  下一步提示:")
        for s2 in steps:
            print("   -", s2)
