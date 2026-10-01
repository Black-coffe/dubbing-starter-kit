"""Baseline for comparison: the same video dubbed end-to-end by the ElevenLabs Dubbing API.

Usage:
    python eldub.py input\\video.mp4

Creates the dub (RU → UK, 1 speaker, voice cloned from the video, no watermark), waits for it,
downloads it and writes output\\<stem>_eldub.m4a. The dubbing_id is kept in
work\\<stem>\\eldub.json, so a rerun resumes or re-downloads instead of paying again.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

from dub import ROOT, ffmpeg, log_timing, lufs, probe_duration, workdir

API = "https://api.elevenlabs.io/v1/dubbing"


def main():
    video = Path(sys.argv[1]).resolve()
    wd = workdir(video)
    state_path = wd / "eldub.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}
    t0 = time.perf_counter()

    if "dubbing_id" not in state:
        with open(video, "rb") as f:
            r = requests.post(API, headers=headers, timeout=600,
                              files={"file": (video.name, f, "video/mp4")},
                              data={"name": f"{video.stem} RU-UK baseline", "source_lang": "ru",
                                    "target_lang": "uk", "num_speakers": "1", "watermark": "false"})
        if r.status_code != 200:
            sys.exit(f"create: HTTP {r.status_code}: {r.text[:500]}")
        state = r.json()
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"created {state['dubbing_id']}, expected ~{state.get('expected_duration_sec', '?')}s")

    dub_id = state["dubbing_id"]
    deadline = time.time() + 30 * 60
    while True:
        r = requests.get(f"{API}/{dub_id}", headers=headers, timeout=60)
        r.raise_for_status()
        meta = r.json()
        if meta.get("status") == "dubbed":
            break
        if meta.get("error") or "fail" in str(meta.get("status")):
            sys.exit(f"dubbing failed: status={meta.get('status')} error={meta.get('error')}")
        if time.time() > deadline:
            sys.exit(f"still {meta.get('status')} after 30 min — rerun later to resume")
        print(f"  status: {meta.get('status')} ({time.perf_counter() - t0:.0f}s)")
        time.sleep(10)
    t_dub = time.perf_counter() - t0

    r = requests.get(f"{API}/{dub_id}/audio/uk", headers=headers, timeout=600)
    if r.status_code != 200:
        sys.exit(f"download: HTTP {r.status_code}: {r.text[:500]}")
    ext = ".mp4" if "mp4" in r.headers.get("content-type", "") else ".mp3"
    raw = wd / f"eldub_raw{ext}"
    raw.write_bytes(r.content)

    out = ROOT / "output" / f"{video.stem}_eldub.m4a"
    out.parent.mkdir(exist_ok=True)
    ffmpeg("-i", str(raw), "-vn", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out))
    log_timing(wd, "eldub", seconds=t_dub, dubbing_id=dub_id, raw=raw.name,
               duration_s=round(probe_duration(out), 2), lufs=lufs(out))
    print(f"dubbed in {t_dub:.0f}s → {out} ({probe_duration(out):.1f}s, {lufs(out):.1f} LUFS)")


if __name__ == "__main__":
    main()
