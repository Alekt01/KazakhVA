# SpeechBrain speaker recognition

This optional service upgrades voice identification from the built-in simple
voiceprint to SpeechBrain ECAPA-TDNN embeddings.

Run it on the Windows/WSL laptop:

```bash
cd ~/myva-training/speaker
source ~/myva-training/.venv/bin/activate
pip install -r requirements.txt
./start_server.sh
```

Start the Mac app with:

```bash
SPEAKER_EMBEDDING_URL=http://127.0.0.1:8766 node server.mjs
```

If the service is offline, the app falls back to the simple local voiceprint
engine. Enroll voices again while ECAPA is enabled so profiles store ECAPA
embeddings.
