# Дорожки A/B/C

Один и тот же фрагмент ролика 3 (https://youtu.be/FA1oVqBTUeM, фрагмент длиной 3:15), дублированный на украинский
тремя способами. Видео без звука, дорожки выровнены по громкости, чтобы громкая не казалась лучше. Слушай и
делай свои выводы. Рядом с каждой дорожкой лежит `.srt` с расшифровкой того, что в ней прозвучало.

| Файл | Вариант | Чем сделана | Как получена |
|---|---|---|---|
| `A.m4a`, `A.srt` | A, наш конвейер | `dub.py`: Demucs, faster-whisper, перевод Claude по `glossary.json`, озвучка ElevenLabs `eleven_v4` клоном голоса, подгонка по времени | `python dub.py run input/video3-fragment.mp4` |
| `B.m4a`, `B.srt` | B, готовый сервис | `eldub.py`: ElevenLabs Dubbing API, одна команда | `python eldub.py input/video3-fragment.mp4`, затем громкость выровнена под A |
| `C.m4a`, `C.srt` | C, всё локально и бесплатно | `variant_local.py` и `tts_omnivoice.py`: перевод в Ollama (MamayLM 12B), голос OmniVoice | `python variant_local.py input/video3-fragment.mp4` |

Лицензия: веса OmniVoice (дорожка C) — CC-BY-NC, только некоммерческое использование. Дорожки сделаны голосом автора
ролика; для своего голоса запускай конвейер на своих материалах (инструкция в `README.md` в корне).

Прямая ссылка на файл после публикации репозитория:
`https://raw.githubusercontent.com/Black-coffe/dubbing-starter-kit/main/tracks/A.m4a`
