"""Handsfree AI Assistant Pipeline.

[Mic] → Porcupine (wakeword) → Google STT → Gemini LLM → gTTS → [Speaker]

Usage:
    uv run python pipeline.py
"""

import os
import subprocess
import tempfile
import time
import wave

import pvporcupine
import speech_recognition as sr
from dotenv import load_dotenv
from google import genai
from gtts import gTTS
from pvrecorder import PvRecorder

load_dotenv()

PV_ACCESS_KEY = os.environ["PV_ACCESS_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MIC_DEVICE_INDEX = 1  # Logitech StreamCam
LANGUAGE = "ja-JP"
WAKEWORD = "computer"
SAMPLE_RATE = 16000
LISTEN_SECONDS = 5


def wait_for_wakeword() -> None:
    """Block until wakeword is detected."""
    porcupine = pvporcupine.create(access_key=PV_ACCESS_KEY, keywords=[WAKEWORD])
    recorder = PvRecorder(
        frame_length=porcupine.frame_length, device_index=MIC_DEVICE_INDEX
    )
    print(f'Waiting for wakeword "{WAKEWORD}"...')
    recorder.start()
    try:
        while True:
            frame = recorder.read()
            if porcupine.process(frame) >= 0:
                print(f"[{time.strftime('%H:%M:%S')}] Wakeword detected!")
                return
    finally:
        recorder.stop()
        recorder.delete()
        porcupine.delete()


def listen_and_transcribe() -> str | None:
    """Record audio after wakeword and transcribe with Google STT."""
    recorder = PvRecorder(frame_length=512, device_index=MIC_DEVICE_INDEX)
    print(f"Listening for {LISTEN_SECONDS} seconds...")
    recorder.start()

    frames: list[bytes] = []
    total = int(SAMPLE_RATE / 512 * LISTEN_SECONDS)
    for _ in range(total):
        frame = recorder.read()
        # Convert to 16-bit PCM bytes
        frames.append(b"".join(v.to_bytes(2, "little", signed=True) for v in frame))

    recorder.stop()
    recorder.delete()

    # Write to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

    # Transcribe with Google STT
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


def ask_gemini(prompt: str, history: list[dict]) -> str:
    """Send prompt to Gemini and return response text."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    messages = history + [{"role": "user", "parts": [{"text": prompt}]}]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config={
            "system_instruction": (
                "You are a helpful voice assistant. "
                "Reply concisely in the same language the user speaks. "
                "Keep responses under 2-3 sentences."
            ),
        },
    )
    text = response.text.strip()
    # Update history
    history.append({"role": "user", "parts": [{"text": prompt}]})
    history.append({"role": "model", "parts": [{"text": text}]})
    # Keep history manageable
    if len(history) > 20:
        history[:] = history[-20:]
    return text


def speak(text: str) -> None:
    """Convert text to speech and play it."""
    lang = "ja" if any("\u3000" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" for c in text) else "en"
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name

    tts = gTTS(text=text, lang=lang)
    tts.save(mp3_path)
    print(f"Assistant: {text}")

    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", mp3_path],
            check=True,
        )
    finally:
        os.unlink(mp3_path)


def main() -> None:
    print("=" * 50)
    print("  Handsfree AI Assistant")
    print(f"  Wakeword: \"{WAKEWORD}\" | Language: {LANGUAGE}")
    print("=" * 50)

    history: list[dict] = []

    while True:
        try:
            wait_for_wakeword()
            text = listen_and_transcribe()
            if text:
                response = ask_gemini(text, history)
                speak(response)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
