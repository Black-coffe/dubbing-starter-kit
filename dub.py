"""Dubbing pipeline (see TASK.md / DECISIONS.md).

Usage:
    python dub.py run input\\video.mp4            # whole pipeline, resumable
    python dub.py run input\\video.mp4 --fresh    # redo audio + transcription too
    python dub.py <step> input\\video.mp4         # one step:
        extract | transcribe | translate | synth | mix | mux

Outputs go to work\\<video stem>\\ and output\\. Each step records its timings in timings.json.
The run stops after translation when new text was machine-translated, so the author can
review phrases_uk.txt; per-phrase fitting decisions live in work\\<stem>\\fit.json.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMUCS_MODEL = "htdemucs"
WHISPER_MODEL = "large-v3"
# Terms Whisper tends to spell in Cyrillic; the prompt nudges it toward Latin spelling.
ASR_PROMPT = "Claude Code, MCP, скиллы, Anthropic, ElevenLabs, API."


class Stop(Exception):
    """A step can't continue without the author (missing translation, review pause)."""


def pick_device(requested: str) -> str:
    """auto → cuda if available, else mps (Apple Silicon), else cpu."""
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def workdir(video: Path) -> Path:
    d = ROOT / "work" / video.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_timing(wd: Path, step: str, **t):
    path = wd / "timings.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[step] = {k: round(v, 2) if isinstance(v, float) else v for k, v in t.items()}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(out.strip())


def lufs(path: Path) -> float:
    """Integrated loudness (EBU R128) of a file."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


# ---------------------------------------------------------------- step 1: audio + separation

def cmd_extract(video: Path, device: str):
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    wd = workdir(video)
    t0 = time.perf_counter()
    # Original track at 48 kHz for the final mix; 44.1 kHz copy is what htdemucs expects.
    ffmpeg("-i", str(video), "-vn", "-ac", "2", "-ar", "48000", str(wd / "audio.wav"))
    ffmpeg("-i", str(video), "-vn", "-ac", "2", "-ar", "44100", str(wd / "demucs_in.wav"))
    t_ffmpeg = time.perf_counter() - t0

    t0 = time.perf_counter()
    device = pick_device(device)
    model = get_model(DEMUCS_MODEL)
    model.to(device).eval()
    sync(device)
    t_load = time.perf_counter() - t0

    wav, sr = sf.read(wd / "demucs_in.wav", dtype="float32", always_2d=True)
    assert sr == model.samplerate, sr
    wav = torch.from_numpy(wav.T.copy())
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav = (wav - mean) / std

    t0 = time.perf_counter()
    with torch.no_grad():
        # shifts=0: no random time offset, so the same video always gives the same stems
        # (and therefore the same transcript and phrase split).
        sources = apply_model(model, wav[None], device=device, shifts=0, split=True,
                              overlap=0.25, progress=False)[0]
    sync(device)
    t_sep = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0

    sources = sources * std + mean
    names = model.sources  # drums, bass, other, vocals
    vocals = sources[names.index("vocals")]
    background = sum(sources[i] for i, n in enumerate(names) if n != "vocals")
    sf.write(wd / "vocals.wav", vocals.cpu().numpy().T, sr, subtype="PCM_16")
    sf.write(wd / "background.wav", background.cpu().numpy().T, sr, subtype="PCM_16")
    (wd / "demucs_in.wav").unlink()

    dur = probe_duration(wd / "audio.wav")
    log_timing(wd, f"extract_{device}", audio_seconds=dur, ffmpeg_s=t_ffmpeg,
               demucs_load_s=t_load, demucs_separate_s=t_sep, realtime_x=dur / t_sep,
               peak_vram_gb=peak_vram)
    print(f"[{device}] audio {dur:.1f}s | ffmpeg {t_ffmpeg:.1f}s | demucs load {t_load:.1f}s | "
          f"separate {t_sep:.1f}s ({dur / t_sep:.0f}x realtime) | peak VRAM {peak_vram:.1f} GB")


# ---------------------------------------------------------------- step 2: transcription

def _add_cuda12_dlls():
    """ctranslate2 on Windows needs cuBLAS/cuDNN 12 DLLs from the nvidia-* wheels."""
    if os.name != "nt":
        return
    import site
    for sp in site.getsitepackages():
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            p = Path(sp) / sub
            if p.is_dir():
                os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]


def cmd_transcribe(video: Path, device: str):
    # CTranslate2 has no Metal/MPS backend, so on a Mac faster-whisper runs on the CPU.
    device = "cpu" if pick_device(device) == "mps" else pick_device(device)
    if device == "cuda":
        _add_cuda12_dlls()
    from faster_whisper import WhisperModel

    wd = workdir(video)
    src = wd / "vocals.wav"
    if not src.exists():
        raise Stop(f"{src} not found — run `extract` first")

    t0 = time.perf_counter()
    model = WhisperModel(WHISPER_MODEL, device=device,
                         compute_type="float16" if device == "cuda" else "int8")
    t_load = time.perf_counter() - t0

    # Decode with ffmpeg ourselves: faster-whisper's PyAV path breaks on current PyAV releases.
    import numpy as np
    pcm = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
                                   "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
    audio = np.frombuffer(pcm, dtype=np.float32)

    t0 = time.perf_counter()
    segments, info = model.transcribe(audio, language="ru", word_timestamps=True,
                                      initial_prompt=ASR_PROMPT)
    segs = []
    for s in segments:  # generator: the actual decoding happens while iterating
        segs.append({
            "id": s.id, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "words": [{"w": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2),
                       "p": round(w.probability, 3)} for w in (s.words or [])],
        })
    t_asr = time.perf_counter() - t0

    (wd / f"transcript_{device}.json" if device != "cuda" else wd / "transcript.json").write_text(
        json.dumps({"model": WHISPER_MODEL, "language": info.language, "segments": segs},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    dur = probe_duration(src)
    n_words = sum(len(s["words"]) for s in segs)
    log_timing(wd, f"transcribe_{device}", audio_seconds=dur, whisper_load_s=t_load, asr_s=t_asr,
               realtime_x=dur / t_asr, segments=len(segs), words=n_words)
    print(f"[{device}] audio {dur:.1f}s | whisper load {t_load:.1f}s | transcribe {t_asr:.1f}s "
          f"({dur / t_asr:.0f}x realtime) | {len(segs)} segments, {n_words} words")


# ---------------------------------------------------------------- step 3: translation

TRANSLATE_MODEL = "claude-sonnet-5-5"
TRANSLATE_EFFORT = "medium"
# $ per 1M tokens for TRANSLATE_MODEL: input, output, cache read, 5-minute cache write.
PRICE = {"in": 2.00, "out": 10.00, "cache_read": 0.20, "cache_write": 2.50}
# Length budget: a phrase may fill this share of the time until the next phrase, at the voice's
# measured pace (chars/s of spoken text). Until the video has takes, the pace of video 3 is used.
BUDGET_FILL = 0.95
DEFAULT_VOICE_RATE = 14.6
SCHEMA = {
    "type": "object",
    "properties": {"segments": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "uk": {"type": "string"}},
        "required": ["n", "uk"], "additionalProperties": False}}},
    "required": ["segments"], "additionalProperties": False,
}


def load_glossary() -> dict:
    return json.loads((ROOT / "glossary.json").read_text(encoding="utf-8"))


def clean_segments(segs: list, g: dict):
    """Apply ASR fixes from the glossary; drop phrases Whisper hallucinates on music."""
    kept, dropped = [], []
    for s in segs:
        text, fixes = s["text"], []
        if any(p.lower() in text.lower() for p in g["drop_phrases"]):
            dropped.append(s)
            continue
        for f in g["asr_fixes"]:
            new = re.sub(f["pattern"], f["replace"], text, flags=re.IGNORECASE)
            if new != text:
                fixes.append(f["replace"])
                text = new
        kept.append({"n": len(kept) + 1, "start": s["start"], "end": s["end"], "ru": text,
                     "ru_asr": s["text"], "fixes": fixes})
    return kept, dropped


def translation_prompt(g: dict) -> str:
    terms = "\n".join(f"- «{t['ru']}» → «{t['uk']}»" + (f" ({t['note']})" if t.get("note") else "")
                      for t in g["terms"])
    return f"""Ты адаптируешь закадровый текст YouTube-ролика про AI в работе с русского на украинский. Текст озвучит клон голоса автора поверх того же видео, поэтому это устная речь, а не письменный перевод, и у каждой фразы жёсткий лимит длины: голос должен успеть договорить до следующей фразы.

Задание приходит в поле task:
- "translate" — адаптируй на украинский каждую фразу segments[].ru;
- "shorten" — фраза segments[].uk длиннее лимита (uk_chars — её текущая длина). Перепиши её до max_chars, сверяясь с оригиналом ru. Если нужно срезать треть и больше — не ищи синонимы, скажи ту же мысль короче и проще.

Лимит. max_chars — сколько символов (с пробелами и знаками) голос успеет произнести за время этой фразы. Не превышай его. Если дословный перевод длиннее — адаптируй: перестрой фразу, возьми слово короче, убери вводные слова и повторы, замени развёрнутую формулировку короткой. Сохраняй смысл, факты и интонацию автора; дословность не нужна. Сокращай сначала связки, вводные слова и повторы; предметы и действия — что именно, кто делает, с чем — оставляй: «Інструкцію і скрипт написала модель» нельзя превращать в «Скрипт написала модель». Но и не режь без нужды: фраза намного короче лимита оставляет в озвучке пустую паузу. Обрубки слов («інак» вместо «інакше») и сокращения («хв», «сек», «т.д.») запрещены: голос прочитает их буквально. Если нормальная фраза в лимит не укладывается — верни самую короткую нормальную, пусть и длиннее лимита: остаток подгонит ускорение.

Не меняются никогда:
- цифры, суммы, время, названия файлов, программ и компаний — ровно как в оригинале;
- эти названия не переводи и не транслитерируй, пиши латиницей ровно так: {", ".join(g["keep"])};
- термины:
{terms}

Язык:
- к зрителю на «ти» (если в оригинале «вы/смотрите» — «ти/дивись»);
- естественный разговорный украинский, без русизмов и канцелярита;
- если в распознанном тексте явная ошибка распознавания, адаптируй то, что автор явно имел в виду.

Одна входная фраза — одна выходная с тем же n. Не объединяй и не разбивай фразы и не переноси смысл в соседнюю: каждая фраза звучит поверх своего куска видео, иначе голос уедет от картинки.

Пример, max_chars 48.
ru: «Сейчас покажу, из чего это всё собрано, полностью, без пропущенных шагов.»
дословно, 69 символов — не влезет: «Зараз покажу, з чого це все зібрано, повністю, без пропущених кроків.»
адаптация, 44 символа: «Покажу, з чого це зібрано, — крок за кроком.»

Пример, task "shorten", max_chars 36.
ru: «Шаблон менять нельзя, это документ компании.»
uk было, 45 символов: «Шаблон міняти не можна, це документ компанії.»
стало, 32 символа: «Шаблон не міняємо: він компанії.»"""


def write_translation(wd: Path, items: list, dropped: list):
    (wd / "translation.json").write_text(json.dumps(
        {"segments": items,
         "dropped": [{"start": d["start"], "end": d["end"], "text": d["text"]} for d in dropped]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    for lang in ("ru", "uk"):
        lines = [f"{i['n']:03d} [{i['start']:6.1f}–{i['end']:6.1f}] {i[lang]}" for i in items]
        (wd / f"phrases_{lang}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_translate(video: Path, device: str, manual: Path | None = None) -> int:
    """Returns how many phrases were newly machine-translated (they need the author's review).

    Already reviewed translations are reused by the Russian text of the phrase, so a
    re-transcription that shifts phrase numbers doesn't attach them to the wrong phrase.
    """
    wd = workdir(video)
    src = wd / "transcript.json"
    if not src.exists():
        raise Stop(f"{src} not found — run `transcribe` first")
    g = load_glossary()
    items, dropped = clean_segments(json.loads(src.read_text(encoding="utf-8"))["segments"], g)

    memory = {}
    if (wd / "translation.json").exists():
        for s in json.loads((wd / "translation.json").read_text(encoding="utf-8"))["segments"]:
            memory[s["ru"]] = (s["uk"], s.get("by", "manual"))
    by_n = {int(k): v for k, v in json.loads(manual.read_text(encoding="utf-8")).items()} if manual else {}
    for i in items:
        if i["n"] in by_n:
            i["uk"], i["by"] = by_n[i["n"]], f"manual ({manual.name})"
        elif i["ru"] in memory:
            i["uk"], i["by"] = memory[i["ru"]]

    rate = voice_rate(wd)
    for i, room in zip(items, rooms(items, probe_duration(wd / "audio.wav"))):
        i["max_chars"] = max(8, int(room * rate * BUDGET_FILL))

    todo = [i for i in items if "uk" not in i]
    t0, usage = time.perf_counter(), {}
    if todo:
        if not has_api_key():
            write_translation(wd, [dict(i, uk=i.get("uk", "")) for i in items], dropped)
            raise Stop(f"{len(todo)} phrases have no translation ({[i['n'] for i in todo]}) and "
                       "ANTHROPIC_API_KEY is not set — set it or pass --manual <file.json>")
        uk, usage = _ask_model("translate", [{"n": i["n"], "max_chars": i["max_chars"], "ru": i["ru"]}
                                             for i in todo], g)
        for i in todo:
            i["uk"], i["by"] = uk[i["n"]], TRANSLATE_MODEL
        usage = [usage]
        # Models count characters poorly: check the length in code and send the long ones back.
        for _ in range(SHORTEN_ROUNDS):
            long = [i for i in items if needs_rework(i["uk"], i["max_chars"])
                    and not i["by"].startswith("manual")]
            if not long:
                break
            uk, u = _ask_model("shorten", [{"n": i["n"], "ru": i["ru"], "uk": i["uk"], "uk_chars": len(i["uk"]),
                                            "max_chars": max(i["max_chars"], int(len(i["uk"]) * MIN_KEEP))}
                                           for i in long], g)
            usage.append(u)
            for i in long:
                i["uk"] = uk[i["n"]]
    t_tr = time.perf_counter() - t0
    glitch = [i["n"] for i in items if glitched(i["uk"])]
    if glitch:
        write_translation(wd, items, dropped)
        raise Stop(f"phrases {glitch} look glitched (another script, JSON scraps) — fix translation.json")
    over = [i["n"] for i in items if len(i["uk"]) > i["max_chars"]]

    write_translation(wd, items, dropped)
    log_timing(wd, "translate", seconds=t_tr, phrases=len(items), reused=len(items) - len(todo),
               machine_translated=len(todo), dropped=len(dropped),
               ru_chars=sum(len(i["ru"]) for i in items), uk_chars=sum(len(i["uk"]) for i in items),
               voice_rate=rate, over_budget=len(over), usage=usage)
    n_manual = len(by_n)
    print(f"{len(items)} phrases: {len(items) - len(todo) - n_manual} reused, {n_manual} from "
          f"{manual.name if manual else '-'}, {len(todo)} translated by {TRANSLATE_MODEL}; "
          f"dropped {len(dropped)}; ASR fixes in {sum(1 for i in items if i['fixes'])} phrases; "
          f"{len(over)} over the length budget at {rate} chars/s")
    return len(todo)


# Model glitches seen on video 3 inside structured output: a letter of another script
# ("Інструкцію й скрипт写ла модель") and a JSON scrap after a double space ("…а нудно.  ','s").
ALLOWED = [(0x0000, 0x02FF), (0x0400, 0x04FF), (0x2000, 0x206F), (0x20AC, 0x20AC), (0x2116, 0x2116)]
FOREIGN = re.compile("[^" + "".join(f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in ALLOWED) + "]")
SCRAP = re.compile(r"['\"],\s*['\"]|\s{2,}|^\s|\s$")


def glitched(text: str) -> bool:
    return bool(FOREIGN.search(text) or SCRAP.search(text))


def needs_rework(text: str, max_chars: int) -> bool:
    return len(text) > max_chars * SHORTEN_OVER


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def voice_rate(wd: Path) -> float:
    """Chars/s the cloned voice speaks, measured by the last `synth` of this video."""
    t = wd / "timings.json"
    rate = json.loads(t.read_text(encoding="utf-8")).get("synth", {}).get("voice_rate") if t.exists() else None
    return rate or DEFAULT_VOICE_RATE


def rooms(items: list, total: float) -> list:
    """Seconds from each phrase's start to the next phrase (or the end of the video)."""
    return [(items[k + 1]["start"] if k + 1 < len(items) else total) - it["start"]
            for k, it in enumerate(items)]


def _ask_model(task: str, segments: list, g: dict) -> tuple[dict, dict]:
    """`translate` or `shorten`; phrases that come back glitched are asked once more, and a second
    glitch stops the run before anything is synthesized."""
    uk, usage = _request(task, segments, g)
    bad = [x for x in segments if glitched(uk[x["n"]])]
    if bad:
        print(f"  glitched phrases {[x['n'] for x in bad]} — asking again")
        again, u2 = _request(task, bad, g)
        uk.update(again)
        usage = {k: round(usage[k] + u2[k], 4) for k in usage}
        still = [x["n"] for x in bad if glitched(uk[x["n"]])]
        if still:
            raise Stop(f"model returned glitched text twice for phrases {still}")
    return uk, usage


def _request(task: str, segments: list, g: dict) -> tuple[dict, dict]:
    """One request. The system prompt is the same for both tasks and marked for caching, so
    the shorten rounds read it from cache."""
    import anthropic

    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=TRANSLATE_MODEL,
        max_tokens=64000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # on a safety decline, the API reruns on a fallback model
        system=[{"type": "text", "text": translation_prompt(g), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user",
                   "content": json.dumps({"task": task, "segments": segments}, ensure_ascii=False)}],
        output_config={"effort": TRANSLATE_EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise Stop(f"model refused: {msg.stop_details}")
    if msg.stop_reason == "max_tokens":
        raise Stop(f"{task} truncated at max_tokens")
    u = msg.usage
    usage = {"in": u.input_tokens, "out": u.output_tokens,
             "cache_read": u.cache_read_input_tokens or 0, "cache_write": u.cache_creation_input_tokens or 0}
    usage["usd"] = round(sum(usage[k] * PRICE[k] for k in PRICE) / 1e6, 4)
    print(f"  {task}: tokens in {usage['in']}, out {usage['out']}, cache read {usage['cache_read']}, "
          f"cache write {usage['cache_write']} ≈ ${usage['usd']}")
    text = next(b.text for b in msg.content if b.type == "text")
    uk = {x["n"]: x["uk"] for x in json.loads(text)["segments"]}
    missing = [x["n"] for x in segments if x["n"] not in uk]
    if missing:
        raise Stop(f"model returned no text for phrases {missing}")
    return uk, usage


# ---------------------------------------------------------------- step 4: voice + placement

VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")  # id of your own cloned voice in ElevenLabs
TTS_MODEL = "eleven_v4"
SR = 48000
# Edge silence below this level is trimmed from each clip before placement.
TRIM = "silenceremove=start_periods=1:start_threshold=-45dB"


def _tts(text: str, prev: str | None, nxt: str | None) -> bytes:
    if not VOICE_ID:
        raise Stop("ELEVENLABS_VOICE_ID is not set (id of your cloned voice)")
    import requests
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        params={"output_format": "mp3_44100_128"},
        json={"text": text, "model_id": TTS_MODEL, "previous_text": prev, "next_text": nxt},
        timeout=120,
    )
    if r.status_code != 200:
        raise Stop(f"ElevenLabs HTTP {r.status_code}: {r.text[:300]}")
    return r.content


def spoken(text: str, g: dict) -> str:
    """Text as sent to TTS: glossary `pronounce` swaps Latin names for how they should sound."""
    for p in g.get("pronounce", []):
        text = text.replace(p["text"], p["say"])
    return text


def take_path(tts_dir: Path, text: str) -> Path:
    """Takes are cached by model + text, so unchanged phrases are never re-synthesized."""
    return tts_dir / (hashlib.sha1(f"{TTS_MODEL}\n{text}".encode()).hexdigest()[:12] + ".wav")


# A take longer than its room by more than SHORTEN_OVER goes back to the model to be shortened,
# at most SHORTEN_ROUNDS times; what still doesn't fit is sped up, at most AUTO_SPEED_MAX.
# The translate step uses the same threshold on text length before any take is synthesized.
AUTO_SPEED_MAX = 1.10
SHORTEN_OVER = AUTO_SPEED_MAX  # up to this much over, a speed-up fits the phrase without cutting words
SHORTEN_ROUNDS = 2
MIN_KEEP = 0.7  # one shorten round cuts at most 30%: deeper cuts broke words ("інак", "хв")
AUTO_SHIFT_GAP = 0.15  # silence kept before a phrase that auto-fitting starts earlier


def cmd_synth(video: Path, device: str):
    import numpy as np
    import soundfile as sf

    wd = workdir(video)
    doc = json.loads((wd / "translation.json").read_text(encoding="utf-8"))
    items = doc["segments"]
    if any(not i["uk"] for i in items):
        raise Stop("translation.json has phrases without Ukrainian text — run `translate` first")
    tts_dir = wd / "tts"
    tts_dir.mkdir(exist_ok=True)
    glitch = [i["n"] for i in items if glitched(i["uk"])]
    if glitch:
        raise Stop(f"phrases {glitch} look glitched (another script, JSON scraps) — fix translation.json first")
    g = load_glossary()
    total = probe_duration(wd / "audio.wav")
    room = rooms(items, total)

    # Author's per-phrase fitting decisions: {"n": {"speed": 1.08, "shift": 0.3, "ru": "..."}}.
    # "ru" pins a decision to its phrase: if re-transcription renumbered phrases, it is skipped.
    # A phrase with a decision is the author's: it is neither shortened nor auto-fitted.
    fit_path = wd / "fit.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8")) if fit_path.exists() else {}
    decisions = {}
    for it in items:
        d = fit.get(str(it["n"])) or {}
        if d.get("ru", it["ru"]) != it["ru"]:
            print(f"  fit.json #{it['n']} is for a different phrase now — skipped")
            d = {}
        decisions[it["n"]] = d

    lengths = {}

    def take_len(text: str) -> float:
        if text not in lengths:
            lengths[text] = sf.info(take_path(tts_dir, text)).duration
        return lengths[text]

    def synth_missing() -> tuple[list, int, int]:
        say = [spoken(it["uk"], g) for it in items]
        n, c = 0, 0
        for k, it in enumerate(items):
            wav = take_path(tts_dir, say[k])
            if wav.exists():
                continue
            mp3 = _tts(say[k], say[k - 1] if k else None, say[k + 1] if k + 1 < len(items) else None)
            wav.with_suffix(".mp3").write_bytes(mp3)
            ffmpeg("-i", str(wav.with_suffix(".mp3")), "-af", f"{TRIM},areverse,{TRIM},areverse",
                   "-ac", "1", "-ar", str(SR), str(wav))
            n += 1
            c += len(say[k])
            print(f"  new take #{it['n']}: {say[k]}")
        return say, n, c

    t0, n_new, chars, usage, shortened = time.perf_counter(), 0, 0, [], set()
    for rnd in range(SHORTEN_ROUNDS + 1):
        say, n, c = synth_missing()
        n_new, chars = n_new + n, chars + c
        long = [k for k, it in enumerate(items)
                if take_len(say[k]) > room[k] * SHORTEN_OVER and not decisions[it["n"]]
                and not it.get("by", "manual").startswith("manual")
                and it.get("shortened", 0) < SHORTEN_ROUNDS]
        if not long or rnd == SHORTEN_ROUNDS or not has_api_key():
            break
        print(f"  round {rnd + 1}: {len(long)} phrases don't fit their time — shortening")
        segs = [{"n": items[k]["n"], "ru": items[k]["ru"], "uk": items[k]["uk"], "uk_chars": len(items[k]["uk"]),
                 "max_chars": max(8, int(len(items[k]["uk"]) * max(MIN_KEEP, room[k] / take_len(say[k]) * BUDGET_FILL)))}
                for k in long]
        uk, u = _ask_model("shorten", segs, g)
        usage.append(u)
        for k in long:
            items[k]["uk"], items[k]["by"] = uk[items[k]["n"]], f"{TRANSLATE_MODEL} (shortened)"
            items[k]["shortened"] = items[k].get("shortened", 0) + 1  # re-runs don't keep cutting
            shortened.add(items[k]["n"])
        write_translation(wd, items, doc.get("dropped", []))
    t_tts = time.perf_counter() - t0
    rates = [len(s) / take_len(s) for s in say if take_len(s) > 1.5]
    rate = round(float(np.median(rates)), 1) if rates else None

    starts = [it["start"] + decisions[it["n"]].get("shift", 0.0) for it in items]
    rows, clips = [], []
    for k, it in enumerate(items):
        take = take_path(tts_dir, say[k])
        raw, _ = sf.read(take, dtype="float32")
        nxt = starts[k + 1] if k + 1 < len(items) else total
        speed, auto = decisions[it["n"]].get("speed"), False
        if speed is None:
            need = len(raw) / SR / (nxt - starts[k])
            speed = min(AUTO_SPEED_MAX, int(need * 100 + 0.999) / 100) if need > 1.0 else 1.0
            auto = speed != 1.0
        if speed != 1.0:
            fitted = take.with_name(f"{take.stem}_x{speed}.wav")
            if not fitted.exists():
                ffmpeg("-i", str(take), "-af", f"rubberband=tempo={speed}:pitchq=quality", str(fitted))
            clip, _ = sf.read(fitted, dtype="float32")
        else:
            clip = raw
        dur = len(clip) / SR
        # Still too long: start earlier into the silence before it, keeping AUTO_SHIFT_GAP of it.
        shift = 0.0
        if not decisions[it["n"]] and starts[k] + dur > nxt:
            prev_end = clips[-1][0] + len(clips[-1][1]) / SR if clips else 0.0
            shift = min(max(starts[k] - prev_end - AUTO_SHIFT_GAP, 0.0), starts[k] + dur - nxt + 0.05)
            starts[k] -= shift
        rows.append({"n": it["n"], "start": round(starts[k], 2), "auto_shift": round(-shift, 2),
                     "orig_s": round(it["end"] - it["start"], 2), "uk_raw_s": round(len(raw) / SR, 2),
                     "speed": speed, "auto_speed": auto, "shortened": it["n"] in shortened,
                     "uk_s": round(dur, 2), "room_s": round(nxt - starts[k], 2),
                     "overlap_s": round(max(0.0, starts[k] + dur - nxt), 2)})
        clips.append((starts[k], clip))
    end = max(total, max(s + len(c) / SR for s, c in clips))
    track = np.zeros(int(end * SR) + 1, dtype=np.float32)
    for start, clip in clips:
        i = int(start * SR)
        track[i:i + len(clip)] += clip
    sf.write(wd / "dub_voice.wav", np.stack([track, track], axis=1), SR, subtype="PCM_16")
    (wd / "timing_report.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    overlaps = [r for r in rows if r["overlap_s"] > 0]
    sped = [r for r in rows if r["auto_speed"]]
    moved = [r for r in rows if r["auto_shift"]]
    log_timing(wd, "synth", model=TTS_MODEL, tts_s=t_tts, new_takes=n_new, chars_sent=chars,
               track_s=round(end, 2), original_s=round(total, 2), overlaps=len(overlaps),
               voice_rate=rate, shortened=len(shortened), auto_sped=len(sped),
               max_auto_speed=max((r["speed"] for r in sped), default=1.0), auto_shifted=len(moved),
               shorten_usage=usage)
    print(f"{n_new} new takes ({chars} chars) in {t_tts:.1f}s; track {end:.1f}s vs original {total:.1f}s; "
          f"voice {rate} chars/s; shortened {len(shortened)}, sped up {len(sped)} "
          f"(≤ {AUTO_SPEED_MAX}), started earlier {len(moved)}")
    for r in overlaps:
        print(f"  overlaps next phrase: #{r['n']} by {r['overlap_s']}s "
              f"(UK {r['uk_s']}s, room {r['room_s']}s) — decide in fit.json")


# ---------------------------------------------------------------- step 5: mix

def cmd_mix(video: Path, device: str):
    """Ukrainian voice at the original voice's loudness, over the background, at the original
    track's loudness, peaks limited to about -1 dBFS, exactly as long as the video."""
    wd = workdir(video)
    t0 = time.perf_counter()
    voice_gain = lufs(wd / "vocals.wav") - lufs(wd / "dub_voice.wav")
    ffmpeg("-i", str(wd / "background.wav"), "-i", str(wd / "dub_voice.wav"), "-filter_complex",
           f"[0]aresample={SR}[b];[1]volume={voice_gain:.2f}dB[v];"
           "[b][v]amix=inputs=2:normalize=0:duration=longest[m]",
           "-map", "[m]", str(wd / "dub_mix_raw.wav"))
    mix_gain = lufs(wd / "audio.wav") - lufs(wd / "dub_mix_raw.wav")
    total = probe_duration(wd / "audio.wav")
    ffmpeg("-i", str(wd / "dub_mix_raw.wav"), "-af",
           f"volume={mix_gain:.2f}dB,alimiter=limit=0.891:level=false,apad",
           "-t", f"{total:.3f}", "-ar", str(SR), str(wd / "dub_uk.wav"))
    (wd / "dub_mix_raw.wav").unlink()
    result = lufs(wd / "dub_uk.wav")
    log_timing(wd, "mix", seconds=time.perf_counter() - t0, voice_gain_db=round(voice_gain, 2),
               mix_gain_db=round(mix_gain, 2), original_lufs=lufs(wd / "audio.wav"), dub_lufs=result)
    print(f"voice {voice_gain:+.1f} dB, mix {mix_gain:+.1f} dB → {result:.1f} LUFS")


# ---------------------------------------------------------------- step 6: mp4

def cmd_mux(video: Path, device: str):
    """MP4 with the Ukrainian track first (default) and the original Russian track second,
    plus the Ukrainian track alone for YouTube Studio → Languages → dub."""
    wd = workdir(video)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    mp4 = out / f"{video.stem}_uk.mp4"
    ffmpeg("-i", str(video), "-i", str(wd / "dub_uk.wav"),
           "-map", "0:v", "-map", "1:a", "-map", "0:a",
           "-c:v", "copy", "-c:a:0", "aac", "-b:a:0", "192k", "-c:a:1", "copy",
           "-metadata:s:a:0", "language=ukr", "-metadata:s:a:0", "title=Українська",
           "-metadata:s:a:1", "language=rus", "-metadata:s:a:1", "title=Русский (оригинал)",
           "-disposition:a:0", "default", "-disposition:a:1", "0",
           "-movflags", "+faststart", str(mp4))
    ffmpeg("-i", str(wd / "dub_uk.wav"), "-c:a", "aac", "-b:a", "192k", str(out / f"{video.stem}_uk.m4a"))
    log_timing(wd, "mux", seconds=time.perf_counter() - t0)
    print(f"{mp4} ({probe_duration(mp4):.1f}s) + {video.stem}_uk.m4a for YouTube Studio")


# ---------------------------------------------------------------- whole pipeline

def cmd_run(video: Path, device: str, manual: Path | None, fresh: bool, no_review: bool):
    wd = workdir(video)
    t_all = time.perf_counter()

    def step(name, fn, *args, skip_if=None):
        if skip_if is not None and skip_if.exists():
            print(f"== {name}: skipped, {skip_if.name} exists (use --fresh to redo)")
            return None
        print(f"== {name}")
        t = time.perf_counter()
        r = fn(video, device, *args)
        print(f"   {name}: {time.perf_counter() - t:.1f}s")
        return r

    step("extract", cmd_extract, skip_if=None if fresh else wd / "vocals.wav")
    step("transcribe", cmd_transcribe, skip_if=None if fresh else wd / "transcript.json")
    new = step("translate", cmd_translate, manual)
    if new and not no_review:
        raise Stop(f"{new} phrases were machine-translated — review {wd / 'phrases_uk.txt'} "
                   "(edit translation.json), then run again")
    step("synth", cmd_synth)
    step("mix", cmd_mix)
    step("mux", cmd_mux)
    total = time.perf_counter() - t_all
    log_timing(wd, "run", seconds=total, fresh=fresh)
    print(f"== done in {total:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "extract", "transcribe", "translate", "synth", "mix", "mux"])
    ap.add_argument("video", type=Path)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--manual", type=Path, help="translate: take {n: uk} from this JSON")
    ap.add_argument("--fresh", action="store_true", help="run: redo extract + transcribe")
    ap.add_argument("--no-review", action="store_true", help="run: don't pause after machine translation")
    a = ap.parse_args()
    video = a.video.resolve()
    try:
        if a.step == "run":
            cmd_run(video, a.device, a.manual, a.fresh, a.no_review)
        elif a.step == "translate":
            cmd_translate(video, a.device, a.manual)
        else:
            {"extract": cmd_extract, "transcribe": cmd_transcribe, "synth": cmd_synth,
             "mix": cmd_mix, "mux": cmd_mux}[a.step](video, a.device)
    except Stop as e:
        sys.exit(f"STOP: {e}")


if __name__ == "__main__":
    main()
