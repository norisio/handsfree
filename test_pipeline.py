"""Automated pipeline test — no human in the loop.

Generates test audio with gTTS, feeds it through STT → Gemini → TTS,
and verifies each stage produces valid output.

Usage:
    uv run python test_pipeline.py
"""

import os
import subprocess
import tempfile
import wave

import speech_recognition as sr
from dotenv import load_dotenv
from google import genai
from gtts import gTTS

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def generate_fixture(text: str, lang: str, filename: str) -> str:
    """Generate a test audio file with gTTS and convert to WAV."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    wav_path = os.path.join(FIXTURES_DIR, filename)
    if os.path.exists(wav_path):
        return wav_path

    mp3_path = wav_path.replace(".wav", ".mp3")
    tts = gTTS(text=text, lang=lang)
    tts.save(mp3_path)

    # Convert mp3 to wav (16kHz mono 16-bit) for SpeechRecognition
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_path],
        capture_output=True,
        check=True,
    )
    os.unlink(mp3_path)
    print(f"  Generated fixture: {wav_path}")
    return wav_path


def test_stt(wav_path: str, language: str) -> str | None:
    """Transcribe a WAV file with Google STT."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return None


def test_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the response."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "system_instruction": (
                "You are a helpful voice assistant. "
                "Reply concisely in the same language the user speaks. "
                "Keep responses under 2-3 sentences."
            ),
        },
    )
    return response.text.strip()


def test_tts(text: str, lang: str) -> str:
    """Generate speech from text, return path to output WAV."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    wav_path = mp3_path.replace(".mp3", ".wav")

    tts = gTTS(text=text, lang=lang)
    tts.save(mp3_path)

    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_path],
        capture_output=True,
        check=True,
    )
    os.unlink(mp3_path)
    return wav_path


TEST_CASES = [
    {"text": "今日の天気はどうですか", "lang": "ja", "stt_lang": "ja-JP", "label": "Japanese: weather question"},
    {"text": "あなたの名前は何ですか", "lang": "ja", "stt_lang": "ja-JP", "label": "Japanese: name question"},
    {"text": "What time is it now", "lang": "en", "stt_lang": "en-US", "label": "English: time question"},
]


def run_tests() -> None:
    print("=" * 60)
    print("  Automated Pipeline Test (no human in the loop)")
    print("=" * 60)

    results = []

    for i, case in enumerate(TEST_CASES):
        print(f"\n--- Test {i + 1}: {case['label']} ---")

        # Step 1: Generate fixture audio
        print("[1/4] Generating test audio...")
        fixture_file = f"test_{i}_{case['lang']}.wav"
        wav_path = generate_fixture(case["text"], case["lang"], fixture_file)

        # Step 2: STT
        print("[2/4] Running STT...")
        transcribed = test_stt(wav_path, case["stt_lang"])
        stt_ok = transcribed is not None and len(transcribed) > 0
        print(f"  Input:      {case['text']}")
        print(f"  Transcribed: {transcribed}")
        print(f"  STT: {'PASS' if stt_ok else 'FAIL'}")

        # Step 3: Gemini LLM
        llm_response = None
        llm_ok = False
        if stt_ok:
            print("[3/4] Querying Gemini...")
            llm_response = test_gemini(transcribed)
            llm_ok = llm_response is not None and len(llm_response) > 0
            print(f"  Response:   {llm_response}")
            print(f"  LLM: {'PASS' if llm_ok else 'FAIL'}")
        else:
            print("[3/4] Skipped (STT failed)")

        # Step 4: TTS
        tts_ok = False
        if llm_ok:
            print("[4/4] Generating TTS response...")
            tts_lang = case["lang"]
            tts_path = test_tts(llm_response, tts_lang)
            # Verify the output WAV is valid and has audio data
            with wave.open(tts_path, "rb") as wf:
                frames = wf.getnframes()
                tts_ok = frames > 0
            print(f"  Output WAV: {frames} frames ({frames / 16000:.1f}s)")
            print(f"  TTS: {'PASS' if tts_ok else 'FAIL'}")
            os.unlink(tts_path)
        else:
            print("[4/4] Skipped (LLM failed)")

        results.append({
            "label": case["label"],
            "stt": stt_ok,
            "llm": llm_ok,
            "tts": tts_ok,
            "all": stt_ok and llm_ok and tts_ok,
        })

    # Summary
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["all"] else "FAIL"
        detail = f"STT:{'OK' if r['stt'] else 'NG'} LLM:{'OK' if r['llm'] else 'NG'} TTS:{'OK' if r['tts'] else 'NG'}"
        print(f"  [{status}] {r['label']}  ({detail})")

    passed = sum(1 for r in results if r["all"])
    print(f"\n  {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("  All stages working!")
    else:
        print("  Some tests failed — check output above.")
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
