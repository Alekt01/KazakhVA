const refs = {
  recordButton: document.querySelector("#connectButton"),
  replayButton: document.querySelector("#muteButton"),
  stopButton: document.querySelector("#disconnectButton"),
  connectionStatus: document.querySelector("#connectionStatus"),
  speechStatus: document.querySelector("#apiStatus"),
  eventStatus: document.querySelector("#eventStatus"),
  conversation: document.querySelector("#conversation"),
  textForm: document.querySelector("#textForm"),
  textInput: document.querySelector("#textInput"),
  profileForm: document.querySelector("#profileForm"),
  workflowForm: document.querySelector("#workflowForm"),
  profiles: document.querySelector("#profiles"),
  workflows: document.querySelector("#workflows"),
  memories: document.querySelector("#memories"),
  learners: document.querySelector("#learners"),
  profileCount: document.querySelector("#profileCount"),
  workflowCount: document.querySelector("#workflowCount"),
  memoryCount: document.querySelector("#memoryCount"),
  learnerCount: document.querySelector("#learnerCount"),
  languageSelect: document.querySelector("#languageSelect"),
  eventLog: document.querySelector("#eventLog"),
  clearLogButton: document.querySelector("#clearLogButton")
};

const languageChoices = new Set(["auto", "en", "ru", "kk"]);
const savedLanguage = localStorage.getItem("voiceAssistantLanguage");

const state = {
  audioContext: null,
  mediaStream: null,
  mediaSource: null,
  processor: null,
  recording: false,
  sampleRate: 16000,
  chunks: [],
  lastReply: "",
  lastReplyLanguage: "auto",
  currentSpeaker: null,
  language: languageChoices.has(savedLanguage) ? savedLanguage : "auto",
  memory: {
    profiles: [],
    workflows: [],
    memories: [],
    learning: {
      learners: []
    }
  },
  speech: null
};

function setStatus(element, text, tone = "neutral") {
  element.textContent = text;
  element.className = `status-pill ${tone}`;
}

function appendMessage(role, text) {
  if (!text.trim()) return null;
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text.trim();
  refs.conversation.append(message);
  refs.conversation.scrollTop = refs.conversation.scrollHeight;
  return message;
}

function appendSources(sources = []) {
  const validSources = sources.filter((source) => source?.title && source?.url);
  if (!validSources.length) return null;
  const message = document.createElement("div");
  message.className = "message sources";
  const heading = document.createElement("strong");
  heading.textContent = "Sources";
  message.append(heading);
  for (const source of validSources) {
    const link = document.createElement("a");
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = `[${source.id}] ${source.title}`;
    message.append(link);
  }
  refs.conversation.append(message);
  refs.conversation.scrollTop = refs.conversation.scrollHeight;
  return message;
}

function appendLog(event) {
  const line = typeof event === "string" ? event : JSON.stringify(event, null, 2);
  refs.eventLog.textContent = `${line}\n\n${refs.eventLog.textContent}`.slice(0, 14000);
}

function selectedLanguage() {
  return languageChoices.has(refs.languageSelect.value) ? refs.languageSelect.value : "auto";
}

function setLanguage(language) {
  state.language = languageChoices.has(language) ? language : "auto";
  refs.languageSelect.value = state.language;
  localStorage.setItem("voiceAssistantLanguage", state.language);
}

function transcriptionLanguageFor(language) {
  return language === "kk" ? "kk" : "auto";
}

function describeError(error) {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  if (error && typeof error === "object") return error.details || error.error || JSON.stringify(error);
  return String(error);
}

function voiceProfileLabel(profile) {
  if (!profile.voice) return "";
  const samples = Number(profile.voice.samples || 1);
  return ` · voice ${samples} sample${samples === 1 ? "" : "s"}`;
}

function memoryMeta(memory) {
  const parts = [memory.kind || "fact"];
  if (memory.speakerName) parts.push(memory.speakerName);
  if (memory.tags?.length) parts.push(memory.tags.join(", "));
  return parts.join(" · ");
}

function languageName(code) {
  if (code === "kk") return "Kazakh";
  if (code === "en") return "English";
  if (code === "ru") return "Russian";
  return code || "Auto";
}

function learnerSummary(learner) {
  const languages = Object.values(learner.languages || {});
  if (!languages.length) return "No lessons yet";
  return languages
    .map((language) => {
      const stats = language.stats || {};
      const words = language.knownWords?.length || 0;
      const weak = language.weakWords?.length || 0;
      const correct = Number(stats.correct || 0);
      const wrong = Number(stats.wrong || 0);
      return `${languageName(language.targetLanguage)} ${language.level || "A1"} · ${words} words · ${weak} weak · ${correct}/${correct + wrong || 0} correct`;
    })
    .join(" | ");
}

async function apiJson(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(describeError(payload));
  return payload;
}

async function apiAudio(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(describeError(errorPayload));
  }
  return response.blob();
}

async function refreshMemory() {
  state.memory = await apiJson("/api/memory");
  refs.profileCount.textContent = String(state.memory.profiles.length);
  refs.workflowCount.textContent = String(state.memory.workflows.length);
  refs.memoryCount.textContent = String((state.memory.memories || []).length);
  refs.learnerCount.textContent = String((state.memory.learning?.learners || []).length);
  refs.profiles.innerHTML = "";
  refs.workflows.innerHTML = "";
  refs.memories.innerHTML = "";
  refs.learners.innerHTML = "";

  for (const profile of state.memory.profiles) {
    const item = document.createElement("div");
    item.className = "item";
    item.innerHTML = `
      <strong>${escapeHtml(profile.name)}</strong>
      <p>${escapeHtml(profile.preferredLanguage || "auto")} · ${escapeHtml(profile.notes || "No notes yet")}${escapeHtml(voiceProfileLabel(profile))}</p>
    `;
    refs.profiles.append(item);
  }

  for (const workflow of state.memory.workflows) {
    const item = document.createElement("div");
    item.className = "item";
    item.innerHTML = `
      <strong>${escapeHtml(workflow.name)}</strong>
      <p>${escapeHtml(workflow.trigger)}</p>
      <p>${escapeHtml(workflow.steps.join(" | "))}</p>
    `;
    refs.workflows.append(item);
  }

  for (const memory of (state.memory.memories || []).slice().reverse()) {
    const item = document.createElement("div");
    item.className = "item";
    item.innerHTML = `
      <strong>${escapeHtml(memoryMeta(memory))}</strong>
      <p>${escapeHtml(memory.text)}</p>
    `;
    refs.memories.append(item);
  }

  for (const learner of state.memory.learning?.learners || []) {
    const active = learner.active?.targetLanguage ? `Active: ${languageName(learner.active.targetLanguage)}` : "Idle";
    const item = document.createElement("div");
    item.className = "item";
    item.innerHTML = `
      <strong>${escapeHtml(learner.name || "Learner")}</strong>
      <p>${escapeHtml(active)}</p>
      <p>${escapeHtml(learnerSummary(learner))}</p>
    `;
    refs.learners.append(item);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function mergeChunks(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += bytesPerSample;
  }

  return new Blob([view], { type: "audio/wav" });
}

function writeString(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

async function startRecording() {
  refs.recordButton.disabled = true;
  refs.stopButton.disabled = false;
  refs.replayButton.disabled = true;
  refs.recordButton.classList.add("recording");
  setStatus(refs.connectionStatus, "Recording", "warn");
  refs.eventStatus.textContent = "Opening microphone";

  try {
    state.chunks = [];
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });
    state.audioContext = new AudioContext({ sampleRate: 16000 });
    state.sampleRate = state.audioContext.sampleRate;
    state.mediaSource = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.processor.onaudioprocess = (event) => {
      if (!state.recording) return;
      const input = event.inputBuffer.getChannelData(0);
      state.chunks.push(new Float32Array(input));
    };

    state.mediaSource.connect(state.processor);
    state.processor.connect(state.audioContext.destination);
    state.recording = true;
    refs.eventStatus.textContent = "Listening";
    appendLog({ recording: "started", sampleRate: state.sampleRate });
  } catch (error) {
    appendMessage("system", `Could not start recording: ${describeError(error)}`);
    appendLog({ recordingError: describeError(error) });
    stopAudioGraph();
    refs.recordButton.disabled = false;
    refs.stopButton.disabled = true;
    refs.recordButton.classList.remove("recording");
    setStatus(refs.connectionStatus, "Mic failed", "bad");
    refs.eventStatus.textContent = "Idle";
  }
}

async function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  refs.stopButton.disabled = true;
  refs.recordButton.classList.remove("recording");
  refs.eventStatus.textContent = "Preparing audio";

  const samples = mergeChunks(state.chunks);
  stopAudioGraph();

  if (samples.length < state.sampleRate * 0.25) {
    appendMessage("system", "Recording was too short. Try holding it a little longer.");
    refs.recordButton.disabled = false;
    setStatus(refs.connectionStatus, "Ready", "good");
    refs.eventStatus.textContent = "Idle";
    return;
  }

  try {
    const wav = encodeWav(samples, state.sampleRate);
    await handleVoiceBlob(wav);
  } catch (error) {
    appendMessage("system", `Voice turn failed: ${describeError(error)}`);
    appendLog({ voiceTurnError: describeError(error) });
  } finally {
    refs.recordButton.disabled = false;
    refs.replayButton.disabled = !state.lastReply;
    setStatus(refs.connectionStatus, "Ready", "good");
    refs.eventStatus.textContent = "Idle";
  }
}

function stopAudioGraph() {
  if (state.processor) {
    state.processor.disconnect();
    state.processor.onaudioprocess = null;
    state.processor = null;
  }
  if (state.mediaSource) {
    state.mediaSource.disconnect();
    state.mediaSource = null;
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }
  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }
}

async function handleVoiceBlob(wav) {
  refs.eventStatus.textContent = "Transcribing locally";
  const replyLanguage = selectedLanguage();
  const listenLanguage = transcriptionLanguageFor(replyLanguage);
  appendLog({ wavBytes: wav.size, replyLanguage, listenLanguage });
  const params = new URLSearchParams({ language: listenLanguage, bias: replyLanguage });
  const transcriptResponse = await fetch(`/api/transcribe?${params.toString()}`, {
    method: "POST",
    headers: {
      "Content-Type": "audio/wav"
    },
    body: wav
  });
  const transcriptPayload = await transcriptResponse.json();
  if (!transcriptResponse.ok) throw new Error(describeError(transcriptPayload));
  const transcript = transcriptPayload.transcript?.trim();
  if (!transcript) {
    appendMessage("system", "Whisper did not hear any words.");
    return;
  }
  if (transcriptPayload.voiceEnrollment) {
    appendMessage(
      "system",
      `Voice enrolled for ${transcriptPayload.voiceEnrollment.name} (${transcriptPayload.voiceEnrollment.samples} sample${transcriptPayload.voiceEnrollment.samples === 1 ? "" : "s"}).`
    );
  } else if (transcriptPayload.speaker?.recognized) {
    const speakerName = transcriptPayload.speaker.name;
    if (state.currentSpeaker !== speakerName) {
      appendMessage("system", `Recognized speaker: ${speakerName} (${transcriptPayload.speaker.confidence}, ${transcriptPayload.speaker.score}).`);
      state.currentSpeaker = speakerName;
    }
  } else if (transcriptPayload.speaker?.candidate) {
    appendLog({ speakerCandidate: transcriptPayload.speaker });
  }
  appendMessage("user", transcript);
  await runLocalAssistantTurn(
    transcript,
    replyLanguage === "auto" ? transcriptPayload.language || "auto" : replyLanguage,
    transcriptPayload.speaker || null
  );
}

async function runLocalAssistantTurn(text, language = selectedLanguage(), speaker = null) {
  refs.eventStatus.textContent = "Thinking locally";
  const response = await apiJson("/api/respond", {
    method: "POST",
    body: JSON.stringify({ text, language, speaker })
  });
  if (response.memory) {
    state.memory = response.memory;
    await refreshMemory();
  }
  if (response.extractedMemories?.length) {
    appendMessage("system", `Saved ${response.extractedMemories.length} long-term memory item${response.extractedMemories.length === 1 ? "" : "s"}.`);
  }
  if (response.mode === "tutor") {
    appendLog({ tutorMode: true });
  }
  if (response.mode === "reminder" && response.reminder) {
    appendLog({ reminderSaved: response.reminder });
  }
  state.lastReply = response.reply;
  state.lastReplyLanguage = response.language || language;
  appendMessage("assistant", response.reply);
  appendSources(response.sources || []);
  await speakText(response.reply, state.lastReplyLanguage);
}

async function speakText(text, language = "auto") {
  if (!text.trim()) return;
  refs.eventStatus.textContent = "Speaking locally";
  const blob = await apiAudio("/api/speak", { text, language });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.addEventListener("ended", () => URL.revokeObjectURL(url), { once: true });
  try {
    await audio.play();
  } catch (error) {
    URL.revokeObjectURL(url);
    appendMessage("system", `Audio playback was blocked. Press Replay to hear it. ${describeError(error)}`);
  }
}

async function sendTextCommand(text) {
  appendMessage("user", text);
  try {
    await runLocalAssistantTurn(text, selectedLanguage());
  } catch (error) {
    appendMessage("system", `Text turn failed: ${describeError(error)}`);
    appendLog({ textTurnError: describeError(error) });
  } finally {
    refs.eventStatus.textContent = "Idle";
  }
}

async function checkHealth() {
  try {
    const health = await apiJson("/api/health");
    state.speech = health.speech;
    const sttReady = Boolean(health.speech?.stt?.ready);
    const ttsReady = Boolean(health.speech?.tts?.ready);
    if (sttReady && ttsReady) {
      const kkFallback = health.speech?.tts?.fallbackVoices?.kk;
      const speechLabel = kkFallback && kkFallback !== "kk" ? "Ready, KK voice fallback" : "Local speech ready";
      setStatus(refs.speechStatus, speechLabel, "good");
      setStatus(refs.connectionStatus, "Ready", "good");
      refs.recordButton.disabled = false;
    } else {
      setStatus(refs.speechStatus, "Run local setup", "warn");
      setStatus(refs.connectionStatus, "Setup needed", "warn");
      refs.recordButton.disabled = true;
    }
    appendLog(health.speech);
  } catch (error) {
    setStatus(refs.speechStatus, "Server offline", "bad");
    setStatus(refs.connectionStatus, "Disconnected", "bad");
    refs.recordButton.disabled = true;
  }
}

async function pollDueReminders() {
  try {
    const payload = await apiJson("/api/reminders/due");
    const dueReminders = payload.reminders || [];
    for (const reminder of dueReminders) {
      const text = `Reminder: ${reminder.text}`;
      appendMessage("assistant", text);
      appendLog({ reminderDue: reminder });
      await speakText(text, selectedLanguage());
    }
  } catch (error) {
    appendLog({ reminderPollError: describeError(error) });
  }
}

refs.recordButton.textContent = "Start recording";
refs.stopButton.textContent = "Stop recording";
refs.replayButton.textContent = "Replay";
setLanguage(state.language);
refs.recordButton.disabled = true;
refs.stopButton.disabled = true;
refs.replayButton.disabled = true;

refs.recordButton.addEventListener("click", startRecording);
refs.stopButton.addEventListener("click", stopRecording);
refs.replayButton.addEventListener("click", () => speakText(state.lastReply, state.lastReplyLanguage));
refs.languageSelect.addEventListener("change", () => {
  setLanguage(selectedLanguage());
  appendLog({ languageMode: state.language });
});
refs.clearLogButton.addEventListener("click", () => {
  refs.eventLog.textContent = "";
});

refs.textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = refs.textInput.value.trim();
  if (!text) return;
  refs.textInput.value = "";
  sendTextCommand(text);
});

refs.profileForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(refs.profileForm);
  await apiJson("/api/profiles", {
    method: "POST",
    body: JSON.stringify(Object.fromEntries(formData.entries()))
  });
  refs.profileForm.reset();
  await refreshMemory();
});

refs.workflowForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(refs.workflowForm);
  await apiJson("/api/workflows", {
    method: "POST",
    body: JSON.stringify(Object.fromEntries(formData.entries()))
  });
  refs.workflowForm.reset();
  await refreshMemory();
});

await checkHealth();
await refreshMemory();
await pollDueReminders();
setInterval(pollDueReminders, 15000);
