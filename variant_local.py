"""Variant C — fully local and free: MamayLM (Ollama) translation + OmniVoice voice clone.

Usage:
    python variant_local.py input\\video.mp4

Needs `dub.py extract` + `transcribe` done (reuses vocals/background/transcript of variant A).
Writes work\\<stem>\\C\\ and output\\<stem>_local.m4a.
NOTE: OmniVoice weights are CC-BY-NC — for listening tests only, not for a monetized channel.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

import dub

OLLAMA = "http://localhost:11434/api/chat"
LLM = "hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q5_K_M"
# Python of the separate OmniVoice venv (see README); set OMNI_PY to its python executable.
OMNI_PY = Path(os.environ.get("OMNI_PY", "omni-env/Scripts/python.exe"))
VOICE_SAMPLE = dub.ROOT / "input" / "voice-sample.wav"


def make_reference(cdir: Path) -> dict:
    """A 6–10 s clip of the author's voice cut at phrase boundaries, plus its transcript."""
    ref_json = cdir / "ref.json"
    if ref_json.exists():
        return json.loads(ref_json.read_text(encoding="utf-8"))
    dub._add_cuda12_dlls()
    from faster_whisper import WhisperModel
    pcm = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(VOICE_SAMPLE),
                                   "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
    segs = list(WhisperModel(dub.WHISPER_MODEL, device="cuda", compute_type="float16")
                .transcribe(np.frombuffer(pcm, dtype=np.float32), language="ru")[0])
    best = None
    for i in range(len(segs)):  # shortest run of whole phrases that reaches 6 s, capped at 10 s
        for j in range(i, len(segs)):
            dur = segs[j].end - segs[i].start
            if dur > 10:
                break
            if dur >= 6:
                best = best or (i, j)
                break
        if best:
            break
    if not best:
        raise SystemExit("no 6–10 s run of whole phrases in voice-sample.wav")
    i, j = best
    start, end = segs[i].start, segs[j].end
    ref = {"ref_audio": str(cdir / "ref.wav"), "start": round(start, 2), "end": round(end, 2),
           "ref_text": " ".join(s.text.strip() for s in segs[i:j + 1])}
    dub.ffmpeg("-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(VOICE_SAMPLE),
               "-ac", "1", "-ar", "24000", ref["ref_audio"])
    ref_json.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    return ref


def translate(items: list, g: dict) -> dict:
    payload = [{"n": i["n"], "duration_s": round(i["end"] - i["start"], 1), "ru": i["ru"]} for i in items]
    schema = {"type": "object", "required": ["segments"], "properties": {"segments": {
        "type": "array", "items": {"type": "object", "required": ["n", "uk"],
                                   "properties": {"n": {"type": "integer"}, "uk": {"type": "string"}}}}}}
    r = requests.post(OLLAMA, timeout=1800, json={
        "model": LLM, "stream": False, "format": schema, "keep_alive": 0,  # unload: free VRAM for TTS
        "options": {"temperature": 0.2, "num_ctx": 8192},
        "messages": [{"role": "system", "content": dub.translation_prompt(g)},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]})
    r.raise_for_status()
    d = r.json()
    uk = {s["n"]: s["uk"].strip() for s in json.loads(d["message"]["content"])["segments"]}
    missing = [i["n"] for i in items if not uk.get(i["n"])]
    if missing:
        raise SystemExit(f"{LLM} returned no translation for phrases {missing}")
    print(f"  tokens: prompt {d.get('prompt_eval_count')}, output {d.get('eval_count')}, "
          f"{d.get('eval_count', 0) / max(d.get('eval_duration', 1) / 1e9, 1e-9):.0f} tok/s")
    return uk


def main():
    video = Path(sys.argv[1]).resolve()
    wd = dub.workdir(video)
    cdir = wd / "C"
    cdir.mkdir(exist_ok=True)
    g = dub.load_glossary()
    timings = {}

    t = time.perf_counter()
    ref = make_reference(cdir)
    timings["reference_s"] = time.perf_counter() - t
    print(f"reference {ref['start']}–{ref['end']}s: {ref['ref_text']}")

    tr_path = cdir / "translation.json"
    items, _ = dub.clean_segments(json.loads((wd / "transcript.json").read_text(encoding="utf-8"))["segments"], g)
    if tr_path.exists():
        items = json.loads(tr_path.read_text(encoding="utf-8"))["segments"]
        print("translation: reused")
    else:
        t = time.perf_counter()
        uk = translate(items, g)
        timings["translate_s"] = time.perf_counter() - t
        for i in items:
            i["uk"] = uk[i["n"]]
        tr_path.write_text(json.dumps({"translator": LLM, "segments": items}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        (cdir / "phrases_uk.txt").write_text(
            "\n".join(f"{i['n']:03d} [{i['start']:6.1f}–{i['end']:6.1f}] {i['uk']}" for i in items) + "\n",
            encoding="utf-8")
        print(f"translation: {LLM} in {timings['translate_s']:.1f}s")

    total = dub.probe_duration(wd / "audio.wav")
    jobs = {"ref_audio": ref["ref_audio"], "ref_text": ref["ref_text"], "out_dir": str(cdir / "tts"),
            "phrases": [{"n": it["n"], "text": dub.spoken(it["uk"], g),
                         "room_s": round((items[k + 1]["start"] if k + 1 < len(items) else total) - it["start"], 2)}
                        for k, it in enumerate(items)]}
    (cdir / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    t = time.perf_counter()
    subprocess.run([str(OMNI_PY), str(dub.ROOT / "tts_omnivoice.py"), str(cdir / "jobs.json")], check=True,
                   env={**os.environ, "PYTHONIOENCODING": "utf-8",
                        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
    timings["tts_s"] = time.perf_counter() - t

    # Place at original start times (no manual fit decisions in this variant).
    track = np.zeros(int(total * dub.SR) + dub.SR, dtype=np.float32)
    for it in items:
        clip, _ = sf.read(cdir / "tts" / f"{it['n']:03d}.wav", dtype="float32")
        i = int(it["start"] * dub.SR)
        track[i:i + len(clip)] += clip[: len(track) - i]
    sf.write(cdir / "dub_voice.wav", np.stack([track, track], axis=1), dub.SR, subtype="PCM_16")

    # Same mix as variant A: voice at the original voice's loudness, whole track at the original's.
    t = time.perf_counter()
    voice_gain = dub.lufs(wd / "vocals.wav") - dub.lufs(cdir / "dub_voice.wav")
    dub.ffmpeg("-i", str(wd / "background.wav"), "-i", str(cdir / "dub_voice.wav"), "-filter_complex",
               f"[0]aresample={dub.SR}[b];[1]volume={voice_gain:.2f}dB[v];"
               "[b][v]amix=inputs=2:normalize=0:duration=longest[m]", "-map", "[m]", str(cdir / "mix_raw.wav"))
    mix_gain = dub.lufs(wd / "audio.wav") - dub.lufs(cdir / "mix_raw.wav")
    out = dub.ROOT / "output" / f"{video.stem}_local.m4a"
    dub.ffmpeg("-i", str(cdir / "mix_raw.wav"), "-af",
               f"volume={mix_gain:.2f}dB,alimiter=limit=0.891:level=false,apad", "-t", f"{total:.3f}",
               "-c:a", "aac", "-b:a", "192k", "-ar", str(dub.SR), str(out))
    (cdir / "mix_raw.wav").unlink()
    timings["mix_s"] = time.perf_counter() - t
    dub.log_timing(wd, "variant_C", **{k: round(v, 1) for k, v in timings.items()},
                   lufs=dub.lufs(out), translator=LLM, tts="OmniVoice")
    print(f"→ {out} ({dub.lufs(out):.1f} LUFS); " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))


if __name__ == "__main__":
    main()
