"""OmniVoice worker for the fully local variant (runs in its own venv).

Usage:
    <omni-venv>/python tts_omnivoice.py jobs.json

jobs.json: {"ref_audio": ..., "ref_text": ..., "out_dir": ...,
            "phrases": [{"n": 1, "text": "...", "room_s": 4.66}, ...]}
Writes <out_dir>/<n>.wav (48 kHz mono, edge silence trimmed) and <out_dir>/report.json.
A phrase that doesn't fit its room is regenerated with OmniVoice's `duration` set to the room.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from omnivoice import OmniVoice

SR_OUT = 48000
GAP = 0.05  # keep a short pause before the next phrase


def trim(x: np.ndarray, sr: int, db: float = -45.0) -> np.ndarray:
    """Cut leading/trailing audio quieter than `db` (10 ms frames)."""
    frame = int(sr * 0.01)
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    loud = np.where(20 * np.log10(rms) > db)[0]
    if len(loud) == 0:
        return x
    return x[loud[0] * frame: (loud[-1] + 1) * frame]


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(jobs["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    t_load = time.perf_counter() - t0
    sr = 24000  # OmniVoice output rate (per README)

    report, t_gen = [], 0.0
    for p in jobs["phrases"]:
        t = time.perf_counter()
        wav = model.generate(text=p["text"], ref_audio=jobs["ref_audio"], ref_text=jobs["ref_text"])[0]
        natural = trim(np.asarray(wav, dtype=np.float32), sr)
        nat_s = len(natural) / sr
        clip, forced = natural, None
        if nat_s > p["room_s"] - GAP:
            forced = round(p["room_s"] - GAP, 2)
            wav = model.generate(text=p["text"], ref_audio=jobs["ref_audio"], ref_text=jobs["ref_text"],
                                 duration=forced)[0]
            clip = trim(np.asarray(wav, dtype=np.float32), sr)
        t_gen += time.perf_counter() - t
        up = AF.resample(torch.from_numpy(clip), sr, SR_OUT).numpy()
        sf.write(out / f"{p['n']:03d}.wav", up, SR_OUT, subtype="PCM_16")
        report.append({"n": p["n"], "natural_s": round(nat_s, 2), "room_s": p["room_s"],
                       "forced_duration_s": forced, "final_s": round(len(clip) / sr, 2),
                       "compression": round(nat_s / (len(clip) / sr), 2) if forced else 1.0})
        print(f"#{p['n']:02d} natural {nat_s:4.1f}s room {p['room_s']:4.1f}s"
              + (f" -> forced {forced}s ({report[-1]['compression']}x)" if forced else ""), flush=True)

    (out / "report.json").write_text(json.dumps(
        {"load_s": round(t_load, 1), "generate_s": round(t_gen, 1),
         "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "phrases": report},
        indent=1), encoding="utf-8")
    print(f"load {t_load:.1f}s, generate {t_gen:.1f}s, peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
