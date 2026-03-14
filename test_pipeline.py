"""Automated pipeline test — no human in the loop.

Generates test audio with gTTS, feeds it through STT → Gemini → TTS,
and verifies each stage produces valid output.

Tests are grouped into conversations where chat history is carried forward,
verifying that the LLM maintains context across turns.

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

SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. "
    "Reply concisely in the same language the user speaks. "
    "Keep responses under 2-3 sentences."
)


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


def test_gemini(prompt: str, history: list[dict]) -> str:
    """Send a prompt to Gemini with chat history and return the response."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    messages = history + [{"role": "user", "parts": [{"text": prompt}]}]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config={"system_instruction": SYSTEM_INSTRUCTION},
    )
    text = response.text.strip()
    # Update history
    history.append({"role": "user", "parts": [{"text": prompt}]})
    history.append({"role": "model", "parts": [{"text": text}]})
    if len(history) > 20:
        history[:] = history[-20:]
    return text


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


# Conversations: each is a list of turns that share chat history.
# Follow-up turns use pronouns/references that only make sense with context.
CONVERSATIONS = [
    {
        "label": "Japanese: multi-turn with context",
        "lang": "ja",
        "stt_lang": "ja-JP",
        "turns": [
            {"text": "富士山の高さを教えてください", "label": "ask about Mt. Fuji height"},
            {"text": "それは世界で何番目に高いですか", "label": "follow-up with 'それ' (it)"},
            {"text": "もう一度最初の質問に答えてください", "label": "ask to repeat first answer"},
        ],
    },
    {
        "label": "English: multi-turn with context",
        "lang": "en",
        "stt_lang": "en-US",
        "turns": [
            {"text": "The capital of France is what", "label": "ask about France capital"},
            {"text": "What language do they speak there", "label": "follow-up with 'there'"},
            {"text": "How do you say thank you in that language", "label": "follow-up with 'that language'"},
        ],
    },
]


def run_single_turn(
    conv_idx: int,
    turn_idx: int,
    turn: dict,
    conv: dict,
    history: list[dict],
) -> dict:
    """Run a single conversation turn through STT → LLM → TTS."""
    label = f"[Conv {conv_idx + 1}, Turn {turn_idx + 1}] {turn['label']}"
    print(f"\n  -- {label} --")

    # Step 1: Generate fixture audio
    fixture_file = f"conv{conv_idx}_turn{turn_idx}_{conv['lang']}.wav"
    wav_path = generate_fixture(turn["text"], conv["lang"], fixture_file)

    # Step 2: STT
    transcribed = test_stt(wav_path, conv["stt_lang"])
    stt_ok = transcribed is not None and len(transcribed) > 0
    print(f"  Input:       {turn['text']}")
    print(f"  Transcribed: {transcribed}")
    print(f"  STT: {'PASS' if stt_ok else 'FAIL'}")

    # Step 3: Gemini LLM (with history)
    llm_response = None
    llm_ok = False
    if stt_ok:
        llm_response = test_gemini(transcribed, history)
        llm_ok = llm_response is not None and len(llm_response) > 0
        print(f"  Response:    {llm_response}")
        print(f"  LLM: {'PASS' if llm_ok else 'FAIL'}")
    else:
        print("  LLM: SKIP (STT failed)")

    # Step 4: TTS
    tts_ok = False
    if llm_ok:
        tts_path = test_tts(llm_response, conv["lang"])
        with wave.open(tts_path, "rb") as wf:
            frames = wf.getnframes()
            tts_ok = frames > 0
        print(f"  TTS: PASS ({frames / 16000:.1f}s)")
        os.unlink(tts_path)
    else:
        print("  TTS: SKIP (LLM failed)")

    return {"label": label, "stt": stt_ok, "llm": llm_ok, "tts": tts_ok, "all": stt_ok and llm_ok and tts_ok}


def run_tests() -> None:
    print("=" * 60)
    print("  Automated Pipeline Test (no human in the loop)")
    print("  Multi-turn conversations with chat history")
    print("=" * 60)

    results = []

    for conv_idx, conv in enumerate(CONVERSATIONS):
        print(f"\n{'─' * 60}")
        print(f"Conversation {conv_idx + 1}: {conv['label']}")
        print(f"{'─' * 60}")

        history: list[dict] = []

        for turn_idx, turn in enumerate(conv["turns"]):
            result = run_single_turn(conv_idx, turn_idx, turn, conv, history)
            results.append(result)

        print(f"\n  History length: {len(history)} messages")

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
