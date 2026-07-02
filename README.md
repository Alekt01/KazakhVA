# Voice AI Assistant

A local-first voice assistant starter that avoids paid speech APIs.

## What is included

- Browser microphone recording to WAV.
- Local speech-to-text with `whisper.cpp`.
- Local text-to-speech with Piper.
- JSON-backed memory for people and learned workflows.
- A simple local rule-based assistant brain, with optional Ollama support.
- Read-only web search for current/online questions.
- Language mode for Auto, English, Russian, and Kazakh.
- Automatic long-term memory extraction when Ollama is enabled.
- Optional SpeechBrain ECAPA-TDNN speaker recognition service.

## One-time local speech setup

```sh
./scripts/setup-local-speech.sh
```

This installs Homebrew `whisper-cpp`, creates `local/piper-venv`, and downloads:

- `ggml-base.bin` for multilingual Whisper transcription.
- `ggml-large-v3-turbo.bin` for better multilingual/Kazakh transcription.
- English Piper voice: `en_US-lessac-medium`.
- Russian Piper voice: `ru_RU-irina-medium`.
- Kazakh Piper voice: `kk_KZ-issai-high`.

The downloaded models live in `local/` and are ignored by Git.

## Optional Kazakh LoRA STT

If the WSL laptop is running the trained Kazakh Whisper LoRA server, start the
app with:

```sh
STT_ENGINE=hybrid-lora LORA_STT_URL=http://127.0.0.1:8765 node server.mjs
```

Then choose `Қазақша` in the app before recording. Auto, English, and Russian
continue to use the normal multilingual `whisper.cpp` path.

If the WSL server is offline, `hybrid-lora` now falls back to local
`whisper.cpp` instead of failing the whole voice turn.

## Languages

Use the language dropdown in the app before recording or typing:

- `Auto` lets Whisper detect the spoken language.
- The assistant listens with Whisper auto-detection so English, Russian, and Kazakh can all be recognized.
- `English`, `Русский`, and `Қазақша` control the assistant's reply language.
- `Қазақша` also enables a guarded Kazakh retry pass for STT when the auto transcript does not look Kazakh enough.
- Kazakh text-to-speech uses the `kk_KZ-issai-high` Piper voice when it has been downloaded by the setup script.
- Saved workflow triggers use fuzzy matching, so small speech-to-text mistakes
  usually still trigger the right workflow.

You can point the app at stronger local models without changing code:

```sh
WHISPER_MODEL=local/models/whisper/ggml-small.bin OLLAMA_MODEL=qwen3:8b node server.mjs
```

To override the Kazakh Piper voice, set:

```sh
PIPER_VOICE_KK=/path/to/kazakh-voice.onnx node server.mjs
```

## Run locally

```sh
node server.mjs
```

If your terminal cannot find `node`, use the Codex runtime path:

```sh
/Users/alekt/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node server.mjs
```

Open:

```text
http://localhost:3000
```

## First things to try

- "Remember me as Alex."
- "Learn workflow: when I say study setup, open my notes, start a 45 minute timer, and play focus music."
- "What do you remember?"

## Optional local AI brain

The default brain is deliberately simple and free. If you install Ollama, you can add a local LLM:

```sh
OLLAMA_MODEL=qwen3:8b node server.mjs
```

No OpenAI key is needed.

## Long-Term Memory

When `OLLAMA_MODEL` is set, the app automatically extracts durable memory after
assistant turns. It saves stable facts, preferences, goals, projects, language
preferences, and standing instructions to `data/store.json`.

It deliberately avoids storing one-off requests, web facts, assistant claims,
passwords, API keys, tokens, payment data, or other secrets. Disable it with:

```sh
MEMORY_EXTRACTION=0 OLLAMA_MODEL=qwen3:8b node server.mjs
```

## Language Tutor

The app includes a first local tutor mode for English and Kazakh. It stores
learning progress in `data/store.json` under `learning`.

Try:

```text
Teach me Kazakh
Quiz me in Kazakh
Practice Kazakh pronunciation
Correct this Kazakh sentence: Мен қазақша үйренем
Show my progress
Stop tutor mode
```

Tutor mode tracks:

- target languages per learner
- introduced words
- weak words
- quiz correct/wrong counts
- active lesson or quiz prompt

The first version ships with a small seed vocabulary in `server.mjs`. Better
teaching materials can be added later as JSON or Markdown lesson packs.

## Web Search

Web search is enabled by default and uses read-only DuckDuckGo HTML results. Ask
things like:

- `search the web for local AI news`
- `what is the latest news about Ollama`
- `интернеттен ... іздеп бер`

The assistant searches, reads short page excerpts, and asks Ollama to answer
with source markers like `[1]`. It does not log in, click buttons, submit forms,
or perform purchases.

The assistant also auto-searches for questions that are likely to need fresh
information, such as weather, prices, sports scores, software versions,
schedules, news, and current office holders like presidents, ministers, mayors,
or CEOs. Stable knowledge questions still use the local Ollama model directly.

To disable web search:

```sh
WEB_SEARCH=0 node server.mjs
```

## Voice Recognition

The app includes a first local voiceprint recognizer. It does not send voice data
to a cloud service. To enroll yourself, record a clear 5-10 second phrase like:

```text
remember my voice as Alexei
```

After that, future recordings are compared with saved voice profiles. If the
match is strong enough, the assistant adds the recognized speaker to the brain
prompt and can use that person's saved profile/preferences.

By default, this uses lightweight audio features, so it is not biometric-grade.
For stronger speaker embeddings, run the SpeechBrain ECAPA service in
`training/speaker` and start the app with:

```sh
SPEAKER_EMBEDDING_URL=http://127.0.0.1:8766 node server.mjs
```

Enroll voices again while ECAPA is enabled so profiles store ECAPA embeddings.
If the service is offline, the app falls back to the built-in local voiceprint
engine.
