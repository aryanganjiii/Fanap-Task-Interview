# RescueHub

Minimal end-to-end demo: two collaborating agents (Fire, Medical) with multi-turn logic,
mock dispatch tool, optional STT/TTS.

## Features
- **Agents**: Fire + Medical (escalation when fire has injuries)
- **Multi-turn**: slot filling for address, injury, severity
- **Mock tool**: `dispatch_resources()` returns simulated dispatch info
- **TTS**: offline via `pyttsx3`
- **STT (optional)**: offline via `vosk` (if model installed)
- **AI model (optional)**: FLAN-T5-Small via `transformers`; falls back to templated replies if missing

## Quick Start (Text I/O, no heavy deps)
```bash
python main.py
```
- Type user's utterances in the console; the system replies and also speaks them (pyttsx3).

## Enable Local AI (optional)
Install optional deps:
```bash
pip install -r requirements.txt
```
Run:
```bash
python main.py --use-ai
```
This uses `google/flan-t5-small` (CPU) to polish agent replies and paraphrase prompts.

## Enable Offline STT (optional)
1) Install `vosk` (already in requirements).
2) Download a small Vosk model (e.g., `vosk-model-small-en-us-0.15`) and unzip somewhere.
3) Run:
```bash
python main.py --stt-model /path/to/vosk-model-small-en-us-0.15
```
Speak through microphone; press Enter on an empty line to end the call.

## Project Layout
```
rescuehub_part2/
  main.py            # entry point, orchestrates a single call
  agents.py          # FireAgent, MedicalAgent, BaseAgent
  tools.py           # mock external tool(s): dispatch_resources
  io_voice.py        # TTS (pyttsx3) + optional STT (vosk)
  nlp.py             # optional AI wrapper (FLAN-T5-small) with graceful fallback
  requirements.txt
  README.md
```

## Notes
- Everything is local/offline-friendly and free.
- If `transformers`/`torch` are heavy to install, you can run without `--use-ai`.
- Replace FLAN-T5 with any local model you already have; the interface is a thin wrapper.
