# Handsfree AI Assistant

Hands-free voice AI assistant running on Raspberry Pi 3B+.
Say the wakeword, speak in Japanese (or English), and get a spoken response.

## Quick Start

```bash
# Set up API keys in .env
cp .env.example .env  # then edit with your keys

# Run the assistant
uv run python pipeline.py
```

Say **"computer"** → speak your question → hear the response. Ctrl+C to quit.

## Pipeline

```
[Mic] → Porcupine (wakeword) → Google STT → Gemini 2.5 Flash → gTTS → [Speaker]
```

### Components

| Stage    | Service                  | Package              | Notes                                       |
|----------|--------------------------|----------------------|---------------------------------------------|
| Wakeword | Porcupine (Picovoice)    | pvporcupine          | On-device, ~3.8% CPU on RPi 3               |
| STT      | Google Web Speech API    | SpeechRecognition    | Japanese (ja-JP) supported, via FLAC upload  |
| LLM      | Gemini API (Google)      | google-genai         | Free tier: 2.5 Flash, no credit card needed  |
| TTS      | Google Text-to-Speech    | gTTS                 | Japanese + English, plays via ffplay         |

### Why this architecture?

- **RPi 3B+ has only 900MB RAM** — local LLM (e.g. picoLLM) is not feasible.
- Porcupine runs on-device for always-on wakeword detection with minimal CPU usage.
- Google Web Speech API chosen over Picovoice Cheetah because **Cheetah doesn't support Japanese**.
- gTTS used instead of Picovoice Orca because Orca only ships with an English model by default.
- Gemini free tier is sufficient for a personal assistant (5-15 RPM, 100-1,000 req/day).

## Hardware

| Item      | Details                                 |
|-----------|-----------------------------------------|
| Board     | Raspberry Pi 3 Model B Plus Rev 1.3     |
| OS        | Debian 13 (trixie), aarch64, kernel 6.12|
| RAM       | ~900 MiB                                |
| Storage   | 58 GB SD card                           |
| Mic       | Logitech StreamCam (USB, device index 1)|
| Audio Out | HDMI / 3.5mm headphone jack             |

## Dev Environment

- Python 3.13.5 (managed with uv 0.10.10)
- Node.js v20.19.2 + npm 9.2.0
- System deps: flac, ffplay (ffmpeg)
- SSH: nao@ras.local

## Required API Keys

Store these in a `.env` file in the project root:

| Variable       | Source                                    |
|----------------|-------------------------------------------|
| PV_ACCESS_KEY  | https://console.picovoice.ai/             |
| GEMINI_API_KEY | https://ai.google.dev/ (Google AI Studio) |
