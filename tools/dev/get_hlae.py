"""下载 HLAE 官方 release 并校验 SHA256（不解压、不安装）。

不走 GitHub API: 那个接口按 IP 限流, 本机已经 403 了。版本号与官方摘要来自
此前对 https://api.github.com/repos/advancedfx/advancedfx/releases/latest
的读取结果 (web_fetch 走的不是这台机器的出口 IP), 直接拼直链下载。
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

TAG = "v2.192.2"
NAME = "hlae_2_192_2.zip"
SIZE = 8997391
SHA256 = "6b020b79edd7dd0ac042677479658ce493c465f578900ef1117a90dafb731add"
URL = f"https://github.com/advancedfx/advancedfx/releases/download/{TAG}/{NAME}"

OUT = Path(r"E:\agent_cs\tools\hlae")


def opener():
    # 本机系统代理 = 127.0.0.1:65532 (注册表 ProxyServer), 直连 github.com 会超时
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"https": "http://127.0.0.1:65532",
                                     "http": "http://127.0.0.1:65532"}))


print(f"版本 : HLAE {TAG}")
print(f"资源 : {NAME}  ({SIZE / 1048576:.1f} MB)")
print(f"官方摘要: sha256:{SHA256}")
print(f"直链 : {URL}\n")

OUT.mkdir(parents=True, exist_ok=True)
dst = OUT / NAME
if dst.is_file() and dst.stat().st_size == SIZE:
    print(f"已存在且大小一致, 跳过下载: {dst}")
else:
    print("下载中 ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "cs2clipper-setup"})
    h = hashlib.sha256()
    got = 0
    with opener().open(req, timeout=300) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            got += len(chunk)
            print(f"\r  {got / 1048576:6.1f} / {SIZE / 1048576:.1f} MB", end="")
    print()
    digest = h.hexdigest()
    print(f"实际摘要: sha256:{digest}")
    if digest != SHA256:
        print("!! 校验失败, 删除文件")
        dst.unlink()
        sys.exit(2)
    print("校验通过")

print(f"\n文件: {dst}  ({dst.stat().st_size / 1048576:.1f} MB)")
