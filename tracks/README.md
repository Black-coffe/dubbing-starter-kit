# Дорожки: оригинал и A/B/C

Один и тот же фрагмент ролика 3 (https://youtu.be/FA1oVqBTUeM, фрагмент длиной 3:15): русский оригинал и три
украинских дубляжа. Дорожки выровнены по громкости (все около -17 LUFS), чтобы громкая не казалась лучше. Слушай и
делай свои выводы: открой `ab.html` в корне, она играет эти файлы и переключает их на лету. Рядом с каждой
дубляжной дорожкой лежит `.srt` с расшифровкой того, что в ней прозвучало.

| Файл | Вариант | Чем сделана | Как получена |
|---|---|---|---|
| `original.m4a` | оригинал, русский | звук ролика 3 канала, тот же фрагмент 3:15 | `ffmpeg -i video3-fragment.mp4 -vn -c:a copy original.m4a` |
| `A.m4a`, `A.srt` | A, наш конвейер | `dub.py`: Demucs, faster-whisper, перевод Claude по `glossary.json`, озвучка ElevenLabs `eleven_v4` клоном голоса, подгонка по времени | `python dub.py run input/video3-fragment.mp4` |
| `B.m4a`, `B.srt` | B, готовый сервис | `eldub.py`: ElevenLabs Dubbing API, одна команда | `python eldub.py input/video3-fragment.mp4`, затем громкость выровнена под A |
| `C.m4a`, `C.srt` | C, всё локально и бесплатно | `variant_local.py` и `tts_omnivoice.py`: перевод в Ollama (MamayLM 12B), голос OmniVoice | `python variant_local.py input/video3-fragment.mp4` |

Лицензия: веса OmniVoice (дорожка C) — CC-BY-NC, только некоммерческое использование. Дорожки сделаны голосом автора
ролика; для своего голоса запускай конвейер на своих материалах (инструкция в `README.md` в корне).

## Файлы и контрольные суммы

Проверка, что скачалось то же самое: `sha256sum tracks/*` (Windows: `certutil -hashfile tracks\A.m4a SHA256`).
Этот `README.md` в список не входит: он описывает сам себя.

| Файл | Байт | SHA-256 |
|---|---|---|
| `A.m4a` | 4806581 | `eee2a929120f75af3ea230090fe92ccf26a475320943400645ee0488b9df519d` |
| `A.srt` | 4776 | `fb50b51c0de003e7fe1cbc5c3a7d492bd4ea8bc56d066f50a52e5d74cad6cc06` |
| `B.m4a` | 4084411 | `7ee0f313ec53d5fc252802fe9d6ab0c86a518b38100fcd1b19d8d0efe91a1ffd` |
| `B.srt` | 5822 | `b52dee15f456470d01b102de82c832937703346ece0687ec7336cb62f7b9906b` |
| `C.m4a` | 4828127 | `fdd72e0fbdc72cc777113681094c46941f50697d698216537a44e3463e6919cc` |
| `C.srt` | 4636 | `1705a2a7920398662e618d3b0462da5f7fa967b725b6236cbde2e3cf5d4a9c22` |
| `original.m4a` | 4792867 | `7e354bf7ffc578cfd7f52b85f5de3ec4c20bad4ba4f392302c70e15a7227c8b6` |

Прямая ссылка на файл после публикации репозитория:
`https://raw.githubusercontent.com/Black-coffe/dubbing-starter-kit/main/tracks/A.m4a`
