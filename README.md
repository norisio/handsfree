# Handsfree AI Assistant

Hands-free voice AI assistant running on Raspberry Pi 3B+.
Say the wakeword, speak in Japanese (or English), and get a spoken response.

## Quick Start

```bash
# Set up API keys in .env
cp .env.example .env  # then edit with your keys

# Run the assistant
uv run python pipeline.py

# Run automated tests (no mic needed)
uv run python test_pipeline.py

# Use a different Gemini model
GEMINI_MODEL=gemini-2.5-flash-lite uv run python test_pipeline.py
```

Say **"ジェミニさん"** -> speak your question -> hear the response. Ctrl+C to quit.

## Pipeline

```
[Mic] -> Porcupine (wakeword) -> Google STT -> Gemini (streaming + MCP tools) -> gTTS -> [Speaker]
```

### Streaming TTS

The LLM response is streamed sentence-by-sentence. TTS synthesis begins as
soon as the first sentence is complete. Sentences are split at delimiters
and played back-to-back via a background thread with a queue.

### MCP Tool Use

MCP (Model Context Protocol) servers are configured in mcp_config.json.
Tools from MCP servers are automatically converted to Gemini function
declarations. When Gemini requests a tool call, it is executed via the
MCP client and the result is sent back for the final response.

Built-in MCP server:
- **datetime** (servers/datetime_server.py) - get_current_time, get_day_of_week

Add more servers by editing mcp_config.json:
```json
{
  "mcpServers": {
    "my_server": {
      "command": "uv",
      "args": ["run", "python", "servers/my_server.py"]
    }
  }
}
```

### Chat History

Chat history is maintained across turns so the assistant can handle follow-up
questions with pronouns and contextual references.

### Components

| Stage    | Service                  | Package              | Notes                                       |
|----------|--------------------------|----------------------|---------------------------------------------|
| Wakeword | Porcupine (Picovoice)    | pvporcupine          | On-device, ~3.8% CPU on RPi 3               |
| STT      | Google Web Speech API    | SpeechRecognition    | Japanese (ja-JP) supported, via FLAC upload  |
| LLM      | Gemini API (Google)      | google-genai         | Streaming, function calling, free tier       |
| TTS      | Google Text-to-Speech    | gTTS                 | Japanese + English, sentence-level streaming  |
| Tools    | MCP                      | mcp                  | Extensible tool use via MCP servers           |

### Why this architecture?

- **RPi 3B+ has only 900MB RAM** - local LLM (e.g. picoLLM) is not feasible.
- Porcupine runs on-device for always-on wakeword detection with minimal CPU.
- Google Web Speech API chosen over Picovoice Cheetah because Cheetah doesn't support Japanese.
- gTTS used instead of Picovoice Orca because Orca only ships with an English model.
- MCP enables extensible tool use without modifying the core pipeline.

## Testing

Automated tests run the full STT -> streaming LLM + MCP -> TTS pipeline without
a microphone or human input. gTTS generates test audio fixtures.

Tests verify:
- **STT** - Google recognizes the generated audio
- **LLM** - Gemini responds with context from chat history
- **Streaming** - response is split into sentences and streamed incrementally
- **TTS** - each sentence chunk produces valid audio output
- **MCP** - tool registration and availability

```bash
uv run python test_pipeline.py
```

Note: Gemini free tier has rate limits (20 req/min per model). The test suite
includes rate limiting (4s between requests). Use GEMINI_MODEL env var to
switch models if needed.

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
- System deps: flac, ffmpeg, libportaudio2
- SSH: nao@ras.local

## Required API Keys

Store these in a .env file in the project root:

| Variable       | Source                                    |
|----------------|-------------------------------------------|
| PV_ACCESS_KEY  | https://console.picovoice.ai/             |
| GEMINI_API_KEY | https://ai.google.dev/ (Google AI Studio) |

Optional:

| Variable      | Default             | Description             |
|---------------|---------------------|-------------------------|
| GEMINI_MODEL  | gemini-2.5-flash    | Gemini model to use     |
| TTS_SPEED     | 1.3                 | Playback speed (1.0 = normal) |
| SILENCE_THRESHOLD | 300             | Audio level below this = silence |
| SILENCE_DURATION  | 1.5             | Seconds of silence to stop recording |

## Project Structure

```
handsfree/
  pipeline.py          # Main assistant loop (wakeword -> STT -> LLM -> TTS)
  test_pipeline.py     # Automated tests (no human needed)
  mcp_client.py        # MCP client + Gemini function calling bridge
  mcp_config.json      # MCP server configuration
  servers/
    datetime_server.py  # Built-in datetime MCP server
  .env                 # API keys (not committed)
```

## Running as a Service

The assistant runs as a systemd user service, starting automatically at boot.

### Install

```bash
mkdir -p ~/.config/systemd/user
cp handsfree.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable handsfree.service
sudo loginctl enable-linger $USER
```

### Manage

```bash
systemctl --user start handsfree       # start
systemctl --user stop handsfree        # stop
systemctl --user restart handsfree     # restart
systemctl --user status handsfree      # check status
journalctl --user -u handsfree -f      # follow logs
```
