# Handsfree AI Assistant

Hands-free voice AI assistant running on Raspberry Pi 3B+.

## Pipeline

```
[Mic] → Porcupine (wakeword) → Google Cloud STT → Gemini API → Orca TTS → [Speaker]
```

### Components

| Stage       | Service                     | Package                | Notes                                    |
|-------------|-----------------------------|------------------------|------------------------------------------|
| Wakeword    | Porcupine (Picovoice)       | `pvporcupine`          | On-device, ~3.8% CPU on RPi 3            |
| STT         | Google Cloud Speech-to-Text | `google-cloud-speech`  | Streaming, Japanese (ja-JP) supported     |
| LLM         | Gemini API (Google)         | `google-genai`         | Free tier: 2.5 Flash/Pro, no credit card  |
| TTS         | Orca (Picovoice)            | `pvorca`               | On-device, low-latency                    |

### Why this architecture?

- **RPi 3B+ has only 900MB RAM** — local LLM is not feasible.
- Porcupine and Orca run on-device (low latency, no cloud dependency for audio).
- Google Cloud STT chosen over Picovoice Cheetah/Leopard because **Cheetah doesn't support Japanese** and Leopard is batch-only.
- Gemini free tier is sufficient for a personal assistant (5-15 RPM, 100-1,000 req/day).

## Hardware

| Item         | Details                                    |
|--------------|--------------------------------------------|
| Board        | Raspberry Pi 3 Model B Plus Rev 1.3        |
| OS           | Debian 13 (trixie), aarch64, kernel 6.12   |
| RAM          | ~900 MiB                                   |
| Storage      | 58 GB SD card                              |
| Microphone   | Logitech StreamCam (USB, device index 1)   |
| Audio Out    | HDMI / 3.5mm headphone jack                |

## Dev Environment

- Python 3.13.5 (managed with uv)
- Node.js v20.19.2 + npm 9.2.0
- SSH: nao@ras.local

## Required API Keys

| Key                 | Source                                          |
|---------------------|-------------------------------------------------|
| Picovoice AccessKey | https://console.picovoice.ai/                  |
| Gemini API Key      | https://ai.google.dev/ (Google AI Studio)       |
| Google Cloud STT    | Google Cloud Console (service account or API key)|
