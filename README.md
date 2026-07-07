# Voice AI Assistant

A local-first voice assistant project that avoids paid speech APIs. It runs a
browser UI on `localhost`, uses local speech models where possible, and stores
assistant data in local JSON files.

## Feature Table

| Feature | Status | Local storage / dependency | Current limits |
| --- | --- | --- | --- |
| Browser voice input | Implemented | Browser microphone, WAV upload to local server | Requires microphone permission and the browser tab open |
| Speech-to-text | Implemented | `whisper.cpp`; optional Kazakh LoRA STT server | Accuracy depends on model/audio; LoRA server is separate |
| Text-to-speech | Implemented | Piper voices in `local/models/piper` | Voice quality depends on installed Piper voice |
| Local assistant brain | Implemented | Rule-based fallback; optional Ollama model | Without Ollama, reasoning is simple |
| Web search | Implemented | DuckDuckGo HTML search, page excerpts | Read-only; no login, clicking, purchasing, or form submission |
| Long-term memory | Implemented | `data/store.json` | Automatic extraction requires Ollama; explicit commands are local |
| Memory control commands | Implemented | `data/store.json` | Personal delete needs recognized speaker or one saved profile |
| Local reminders | Implemented | `data/reminders.json` | In-app only; no macOS/Windows system notification daemon |
| Language tutor | Implemented, early version | `data/store.json` under `learning` | Seed English/Kazakh lessons only, not a full curriculum |
| Speaker recognition | Implemented, experimental | Local voiceprint or optional SpeechBrain ECAPA service | Not biometric-grade authentication |
| Kazakh STT training scripts | Present | `training/whisper` | Training is manual and hardware-dependent |

## Architecture

```text
Browser UI
  public/index.html
  public/app.js
  public/styles.css
        |
        | localhost HTTP API
        v
Node server
  server.mjs
        |
        +-- STT: whisper.cpp or optional LoRA STT server
        +-- TTS: Piper
        +-- Brain: local rules or optional Ollama
        +-- Web search: read-only DuckDuckGo HTML
        +-- Speaker recognition: local voiceprint or optional ECAPA server
        |
        +-- data/store.json       people, workflows, memories, tutor progress
        +-- data/reminders.json   local timers and reminders
```

Main modules:

- [server.mjs](/Users/alekt/Documents/MyVA/server.mjs): HTTP server, local assistant flow, speech endpoints, web search, tutor mode, reminders.
- [lib/memory.mjs](/Users/alekt/Documents/MyVA/lib/memory.mjs): local memory store helpers.
- [lib/reminders.mjs](/Users/alekt/Documents/MyVA/lib/reminders.mjs): reminder parsing and JSON store helpers.
- [training/whisper](/Users/alekt/Documents/MyVA/training/whisper): Kazakh Whisper LoRA training/server scripts.
- [training/speaker](/Users/alekt/Documents/MyVA/training/speaker): optional SpeechBrain ECAPA speaker embedding service.

## Setup Steps

1. Install or use Node.js.

   If `node` is available:

   ```sh
   node --version
   ```

   If your shell cannot find Node, this project has been run with the Codex
   runtime path:

   ```sh
   /Users/alekt/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node server.mjs
   ```

2. Install local speech tools and models.

   ```sh
   ./scripts/setup-local-speech.sh
   ```

   This installs or prepares:

   - `whisper.cpp`
   - `ggml-base.bin`
   - `ggml-large-v3-turbo.bin`
   - Piper runtime
   - English Piper voice: `en_US-lessac-medium`
   - Russian Piper voice: `ru_RU-irina-medium`
   - Kazakh Piper voice: `kk_KZ-issai-high`

   Downloaded models live in `local/` and are ignored by Git.

3. Start the app.

   Basic local mode:

   ```sh
   node server.mjs
   ```

   With Ollama:

   ```sh
   OLLAMA_MODEL=qwen3:8b node server.mjs
   ```

   With optional Kazakh LoRA STT server:

   ```sh
   STT_ENGINE=hybrid-lora LORA_STT_URL=http://127.0.0.1:8765 node server.mjs
   ```

   With optional SpeechBrain ECAPA speaker service:

   ```sh
   SPEAKER_EMBEDDING_URL=http://127.0.0.1:8766 node server.mjs
   ```

   Combined local setup example:

   ```sh
   STT_ENGINE=hybrid-lora \
   LORA_STT_URL=http://127.0.0.1:8765 \
   OLLAMA_MODEL=qwen3:8b \
   SPEAKER_EMBEDDING_URL=http://127.0.0.1:8766 \
   node server.mjs
   ```

4. Open the browser app.

   ```text
   http://localhost:3000
   ```

5. Run tests.

   ```sh
   node --test
   ```

   If `npm` is available, this also works:

   ```sh
   npm test
   ```

## Environment Options

The server reads environment variables from the shell. It does not automatically
load `.env` files.

Useful options:

| Variable | Purpose |
| --- | --- |
| `PORT` | HTTP port, default `3000` |
| `WHISPER_CPP_BIN` | Path to `whisper-cli` |
| `WHISPER_MODEL` | Path to a `whisper.cpp` model |
| `STT_ENGINE` | `whisper.cpp`, `lora`, or `hybrid-lora` |
| `LORA_STT_URL` | Optional local/WSL Kazakh LoRA STT server URL |
| `PIPER_BIN` | Path to Piper executable |
| `PIPER_VOICE_EN` | English Piper voice path |
| `PIPER_VOICE_RU` | Russian Piper voice path |
| `PIPER_VOICE_KK` | Kazakh Piper voice path |
| `OLLAMA_MODEL` | Enables Ollama responses and model memory extraction |
| `OLLAMA_URL` | Ollama server URL, default `http://127.0.0.1:11434` |
| `WEB_SEARCH` | Set `0` to disable web search |
| `MEMORY_EXTRACTION` | Set `0` to disable automatic memory extraction |
| `SPEAKER_EMBEDDING_URL` | Optional ECAPA speaker embedding service URL |

## First Things To Try

```text
Remember me as Alexei
remember this: I prefer short answers.
what do you remember about me
set timer for 5 minutes
remind me tomorrow at 5 to study Kazakh
Teach me Kazakh
Quiz me in Kazakh
search the web for local AI news
remember my voice as Alexei
```

## Local Memory

The main local memory file is `data/store.json`. It stores:

- saved people
- learned workflows
- long-term memory items
- language tutor progress
- voice profile metadata and embeddings

Explicit memory commands work without Ollama:

```text
remember this: I prefer short answers.
forget this: I prefer short answers.
what do you remember about me
delete my memory
```

Automatic long-term memory extraction only runs when `OLLAMA_MODEL` is set and
`MEMORY_EXTRACTION` is not `0`. It tries to save stable facts, preferences,
goals, projects, language preferences, and standing instructions.

The extractor is instructed not to save passwords, API keys, tokens, payment
data, or other secrets, but you should still avoid saying secrets to the app.

## Local Reminders

Reminders are stored in `data/reminders.json`.

Supported examples:

```text
set timer for 10 minutes
set a 5 minute timer to check tea
remind me tomorrow at 5 to call mom
remind me tomorrow at 17:30 to study Kazakh
```

If you say `tomorrow at 5` without `am` or `pm`, the parser treats it as 17:00
local time. Use `5 am` for morning reminders.

Current limitation: reminders are delivered inside the browser app. The page
polls `/api/reminders/due` while it is open and speaks due reminders with Piper.
There is no system notification daemon yet.

## Languages

Use the language dropdown before recording or typing.

- `Auto` lets Whisper detect speech language.
- `English`, `Русский`, and `Қазақша` control the assistant reply language.
- Kazakh TTS uses `kk_KZ-issai-high` when installed.
- `hybrid-lora` can use a local Kazakh LoRA STT server for Kazakh-biased transcription and fall back to `whisper.cpp` if configured that way.

You can override the Kazakh Piper voice:

```sh
PIPER_VOICE_KK=/path/to/kazakh-voice.onnx node server.mjs
```

## Language Tutor

Tutor mode currently supports English and Kazakh with a small seed vocabulary in
`server.mjs`. It tracks introduced words, weak words, quiz score, and active
lesson/quiz state in `data/store.json`.

Try:

```text
Teach me Kazakh
Quiz me in Kazakh
Practice Kazakh pronunciation
Correct this Kazakh sentence: Мен қазақша үйренем
Show my progress
Stop tutor mode
```

This is not yet a full language-learning system. Better lesson packs should be
added as structured JSON or Markdown materials.

## Web Search

Web search is enabled by default through read-only DuckDuckGo HTML results. The
assistant can fetch short page excerpts and ask Ollama to answer from those
results.

Examples:

```text
search the web for local AI news
what is the latest news about Ollama
интернеттен жаңалықтарды іздеп бер
```

The app does not log in to websites, click buttons, submit forms, or perform
purchases. Disable web search with:

```sh
WEB_SEARCH=0 node server.mjs
```

## Voice Recognition

The app can enroll a local voice profile from a recorded phrase:

```text
remember my voice as Alexei
```

By default, this uses a lightweight local voiceprint. For stronger embeddings,
run the optional SpeechBrain ECAPA service in `training/speaker` and start the
app with:

```sh
SPEAKER_EMBEDDING_URL=http://127.0.0.1:8766 node server.mjs
```

Enroll voices again while ECAPA is enabled so profiles store ECAPA embeddings.
This feature is useful for personalization, but it should not be treated as
secure identity verification.

## Privacy

Local by default:

- Browser audio is sent to the local Node server on `localhost`.
- Speech-to-text uses local `whisper.cpp` unless you configure a local/WSL LoRA STT server.
- Text-to-speech uses local Piper.
- People, workflows, memories, tutor progress, and voice profiles are stored in `data/store.json`.
- Timers and reminders are stored in `data/reminders.json`.
- `data/store.json`, `data/reminders.json`, `local/`, and `data/tmp/` are ignored by Git.

Optional network behavior:

- Web search contacts DuckDuckGo and fetched result pages.
- Ollama calls go to `OLLAMA_URL`, normally `http://127.0.0.1:11434`.
- The optional SpeechBrain/LoRA services may run on another local machine if you point the app there.

Secrets:

- Do not commit `.env`.
- Do not store API keys, passwords, tokens, or payment details in memory.
- The app is a local project assistant, not a hardened secrets manager.

## Roadmap

Near-term:

- Add a visible reminders panel with edit/delete controls.
- Add structured language lesson packs outside `server.mjs`.
- Add tests for reminder API behavior and memory command handling.
- Add import/export tools for local memory.
- Improve Kazakh STT evaluation with a repeatable test set.

Later:

- Background reminder worker or OS notifications.
- Better workflow execution with explicit permission gates.
- More robust multilingual tutor feedback.
- Speaker profile management UI.
- Safer configuration loading and startup diagnostics.

Not planned yet:

- Cloud account sync.
- Website automation that logs in or performs purchases.
- Treating voice recognition as security authentication.
