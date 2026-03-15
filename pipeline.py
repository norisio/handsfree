"""Handsfree AI Assistant Pipeline.

[Mic] → Porcupine (wakeword) → Google STT → Gemini LLM (streaming + MCP tools) → gTTS → [Speaker]

LLM response is streamed sentence-by-sentence: TTS synthesis and playback
begin as soon as the first sentence is ready, without waiting for the full
response. MCP tool calls are resolved before streaming begins.

Usage:
    uv run python pipeline.py
"""

import asyncio
import os
import subprocess
import tempfile
import threading
import time
import wave
from queue import Queue

import pvporcupine
import speech_recognition as sr
from dotenv import load_dotenv
from google import genai
from gtts import gTTS
from pvrecorder import PvRecorder

from mcp_client import McpManager

load_dotenv()

PV_ACCESS_KEY = os.environ["PV_ACCESS_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MIC_DEVICE_INDEX = 1  # Logitech StreamCam
LANGUAGE = "ja-JP"
WAKEWORD_PATH = os.environ.get(
    "WAKEWORD_PATH",
    os.path.join(os.path.dirname(__file__), "ジェミニさん_ja_raspberry-pi_v4_0_0.ppn"),
)
WAKEWORD_LABEL = "ジェミニさん"
SAMPLE_RATE = 16000
MAX_LISTEN_SECONDS = 15
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.3"))  # 1.0 = normal, 1.5 = 50% faster
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
SFX_DETECTED = os.path.join(ASSETS_DIR, "detected.mp3")
SFX_RECORDED = os.path.join(ASSETS_DIR, "recorded.mp3")

# Silence detection
SILENCE_THRESHOLD = int(os.environ.get("SILENCE_THRESHOLD", "300"))  # audio level below this = silence
SILENCE_DURATION = float(os.environ.get("SILENCE_DURATION", "1.5"))  # seconds of silence to stop

SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. "
    "Reply concisely in the same language the user speaks. "
    "Keep responses under 2-3 sentences. "
    "Use available tools when they can help answer the user's question."
)

SENTENCE_DELIMITERS = set("。！？.!?\n")


def play_sfx(path: str) -> None:
    """Play a sound effect (non-blocking)."""
    if os.path.exists(path):
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def detect_lang(text: str) -> str:
    """Detect language for TTS based on character content."""
    if any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in text):
        return "ja"
    return "en"


def wait_for_wakeword() -> None:
    """Block until wakeword is detected."""
    porcupine = pvporcupine.create(
        access_key=PV_ACCESS_KEY,
        keyword_paths=[WAKEWORD_PATH],
        model_path=os.path.join(ASSETS_DIR, "porcupine_params_ja.pv"),
    )
    recorder = PvRecorder(
        frame_length=porcupine.frame_length, device_index=MIC_DEVICE_INDEX
    )
    print(f'Waiting for wakeword "{WAKEWORD_LABEL}"...')
    recorder.start()
    try:
        while True:
            frame = recorder.read()
            if porcupine.process(frame) >= 0:
                print(f"[{time.strftime('%H:%M:%S')}] Wakeword detected!")
                play_sfx(SFX_DETECTED)
                return
    finally:
        recorder.stop()
        recorder.delete()
        porcupine.delete()


def listen_and_transcribe() -> str | None:
    """Record audio after wakeword, stop on silence, transcribe with Google STT."""
    recorder = PvRecorder(frame_length=512, device_index=MIC_DEVICE_INDEX)
    print("Listening... (speak now, stops on silence)")
    recorder.start()

    frames: list[bytes] = []
    max_frames = int(SAMPLE_RATE / 512 * MAX_LISTEN_SECONDS)
    silence_frames_needed = int(SAMPLE_RATE / 512 * SILENCE_DURATION)
    consecutive_silent = 0
    has_speech = False

    for _ in range(max_frames):
        frame = recorder.read()
        frames.append(b"".join(v.to_bytes(2, "little", signed=True) for v in frame))

        level = max(abs(v) for v in frame)
        if level > SILENCE_THRESHOLD:
            has_speech = True
            consecutive_silent = 0
        else:
            consecutive_silent += 1

        # Stop after sustained silence, but only if we heard speech first
        if has_speech and consecutive_silent >= silence_frames_needed:
            break

    recorder.stop()
    recorder.delete()

    if not has_speech:
        print("No speech detected.")
        return None

    play_sfx(SFX_RECORDED)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=LANGUAGE)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"STT error: {e}")
        return None
    finally:
        os.unlink(wav_path)


def _tts_worker(audio_queue: Queue) -> None:
    """Background thread: pull audio file paths from queue and play them."""
    while True:
        item = audio_queue.get()
        if item is None:  # poison pill
            break
        mp3_path = item
        try:
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if TTS_SPEED != 1.0:
                cmd += ["-af", f"atempo={TTS_SPEED}"]
            cmd.append(mp3_path)
            subprocess.run(cmd, check=True)
        finally:
            os.unlink(mp3_path)
        audio_queue.task_done()


def _synthesize_sentence(sentence: str, lang: str, audio_queue: Queue) -> None:
    """Synthesize one sentence and enqueue the audio for playback."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    tts = gTTS(text=sentence, lang=lang)
    tts.save(mp3_path)
    audio_queue.put(mp3_path)


def stream_and_speak(
    prompt: str,
    history: list[dict],
    mcp: McpManager,
    loop: asyncio.AbstractEventLoop,
) -> str:
    """Stream Gemini response with MCP tool support, TTS per sentence."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    messages = history + [{"role": "user", "parts": [{"text": prompt}]}]

    # Audio playback queue and worker thread
    audio_queue: Queue = Queue()
    player = threading.Thread(target=_tts_worker, args=(audio_queue,), daemon=True)
    player.start()

    full_response = []
    sentence_buf = []

    def flush_sentence():
        sentence = "".join(sentence_buf).strip()
        sentence_buf.clear()
        if sentence:
            lang = detect_lang(sentence)
            print(f"  >> {sentence}")
            _synthesize_sentence(sentence, lang, audio_queue)

    # Import here to avoid circular
    from mcp_client import gemini_stream_with_tools

    for text_chunk in gemini_stream_with_tools(
        client=client,
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=messages,
        system_instruction=SYSTEM_INSTRUCTION,
        mcp=mcp,
        loop=loop,
    ):
        full_response.append(text_chunk)
        for char in text_chunk:
            sentence_buf.append(char)
            if char in SENTENCE_DELIMITERS:
                flush_sentence()

    # Flush any remaining text
    flush_sentence()

    # Wait for all audio to finish playing
    audio_queue.join()
    audio_queue.put(None)  # stop worker
    player.join()

    full_text = "".join(full_response).strip()

    # Update history
    history.append({"role": "user", "parts": [{"text": prompt}]})
    history.append({"role": "model", "parts": [{"text": full_text}]})
    if len(history) > 20:
        history[:] = history[-20:]

    return full_text


def main() -> None:
    print("=" * 50)
    print("  Handsfree AI Assistant")
    print(f"  Wakeword: \"{WAKEWORD_LABEL}\" | Language: {LANGUAGE}")
    print("  Streaming TTS + MCP tools enabled")
    print("=" * 50)

    # Set up async event loop for MCP
    loop = asyncio.new_event_loop()

    # Connect to MCP servers
    mcp = McpManager()
    print("Connecting to MCP servers...")
    loop.run_until_complete(mcp.connect())

    history: list[dict] = []

    try:
        while True:
            try:
                wait_for_wakeword()
                text = listen_and_transcribe()
                if text:
                    print("Assistant:")
                    stream_and_speak(text, history, mcp, loop)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    finally:
        loop.run_until_complete(mcp.close())
        loop.close()


if __name__ == "__main__":
    main()
