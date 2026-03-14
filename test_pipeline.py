"""Automated pipeline test — no human in the loop.

Generates test audio with gTTS, feeds it through STT → Gemini (streaming) → TTS,
and verifies each stage produces valid output.

Tests are grouped into conversations where chat history is carried forward,
verifying that the LLM maintains context across turns.

The LLM stage uses streaming to verify sentence-by-sentence TTS synthesis
works correctly (matching pipeline.py's streaming behavior).

Usage:
    uv run python test_pipeline.py
"""

import os
import subprocess
import tempfile
import threading
import time
import wave
from queue import Queue

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

SENTENCE_DELIMITERS = set("。！？.!?\n")


def detect_lang(text: str) -> str:
    """Detect language for TTS based on character content."""
    if any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in text):
        return "ja"
    return "en"


def generate_fixture(text: str, lang: str, filename: str) -> str:
    """Generate a test audio file with gTTS and convert to WAV."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    wav_path = os.path.join(FIXTURES_DIR, filename)
    if os.path.exists(wav_path):
        return wav_path

    mp3_path = wav_path.replace(".wav", ".mp3")
    tts = gTTS(text=text, lang=lang)
    tts.save(mp3_path)

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


def _tts_worker(audio_queue: Queue, results: list[dict]) -> None:
    """Background thread: synthesize and verify TTS audio from queue."""
    while True:
        item = audio_queue.get()
        if item is None:
            break
        sentence, lang = item
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        wav_path = mp3_path.replace(".mp3", ".wav")

        tts = gTTS(text=sentence, lang=lang)
        tts.save(mp3_path)

        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True,
            check=True,
        )
        os.unlink(mp3_path)

        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
        os.unlink(wav_path)

        results.append({"sentence": sentence, "frames": frames, "ok": frames > 0})
        audio_queue.task_done()


def test_streaming_gemini_tts(prompt: str, history: list[dict]) -> dict:
    """Stream Gemini response, synthesize TTS per sentence, verify each chunk.

    Returns dict with full_text, sentences list, and per-sentence TTS results.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    messages = history + [{"role": "user", "parts": [{"text": prompt}]}]

    # TTS verification queue and worker
    audio_queue: Queue = Queue()
    tts_results: list[dict] = []
    worker = threading.Thread(
        target=_tts_worker, args=(audio_queue, tts_results), daemon=True
    )
    worker.start()

    full_response = []
    sentence_buf = []
    sentences_sent = 0
    t_first_sentence = None
    t_start = time.monotonic()

    def flush_sentence():
        nonlocal sentences_sent, t_first_sentence
        sentence = "".join(sentence_buf).strip()
        sentence_buf.clear()
        if sentence:
            if t_first_sentence is None:
                t_first_sentence = time.monotonic()
            lang = detect_lang(sentence)
            audio_queue.put((sentence, lang))
            sentences_sent += 1

    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=messages,
        config={"system_instruction": SYSTEM_INSTRUCTION},
    ):
        text = chunk.text or ""
        full_response.append(text)
        for char in text:
            sentence_buf.append(char)
            if char in SENTENCE_DELIMITERS:
                flush_sentence()

    # Flush remaining
    flush_sentence()

    # Wait for TTS worker to finish
    audio_queue.join()
    audio_queue.put(None)
    worker.join()

    t_end = time.monotonic()
    full_text = "".join(full_response).strip()

    # Update history
    history.append({"role": "user", "parts": [{"text": prompt}]})
    history.append({"role": "model", "parts": [{"text": full_text}]})
    if len(history) > 20:
        history[:] = history[-20:]

    return {
        "full_text": full_text,
        "sentences_sent": sentences_sent,
        "tts_results": tts_results,
        "time_to_first_sentence": (t_first_sentence - t_start) if t_first_sentence else None,
        "total_time": t_end - t_start,
    }


# Conversations: each is a list of turns that share chat history.
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
    """Run a single conversation turn through STT → streaming LLM → TTS."""
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

    # Step 3+4: Streaming LLM + TTS (with history)
    llm_ok = False
    tts_ok = False
    streaming_ok = False
    if stt_ok:
        result = test_streaming_gemini_tts(transcribed, history)

        llm_ok = bool(result["full_text"])
        print(f"  Response:    {result['full_text']}")
        print(f"  LLM: {'PASS' if llm_ok else 'FAIL'}")

        # Verify streaming: sentences were sent incrementally
        n_sentences = result["sentences_sent"]
        tts_all_ok = all(r["ok"] for r in result["tts_results"])
        tts_ok = len(result["tts_results"]) > 0 and tts_all_ok
        streaming_ok = n_sentences > 0

        for r in result["tts_results"]:
            print(f"    TTS chunk: \"{r['sentence']}\" → {r['frames']} frames {'OK' if r['ok'] else 'NG'}")

        if result["time_to_first_sentence"] is not None:
            print(f"  Time to first sentence: {result['time_to_first_sentence']:.2f}s")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Sentences streamed: {n_sentences}")
        print(f"  Streaming: {'PASS' if streaming_ok else 'FAIL'}")
        print(f"  TTS: {'PASS' if tts_ok else 'FAIL'}")
    else:
        print("  LLM+TTS: SKIP (STT failed)")

    return {
        "label": label,
        "stt": stt_ok,
        "llm": llm_ok,
        "tts": tts_ok,
        "streaming": streaming_ok,
        "all": stt_ok and llm_ok and tts_ok and streaming_ok,
    }


def run_tests() -> None:
    print("=" * 60)
    print("  Automated Pipeline Test (no human in the loop)")
    print("  Streaming LLM + sentence-by-sentence TTS")
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
        detail = (
            f"STT:{'OK' if r['stt'] else 'NG'} "
            f"LLM:{'OK' if r['llm'] else 'NG'} "
            f"Stream:{'OK' if r['streaming'] else 'NG'} "
            f"TTS:{'OK' if r['tts'] else 'NG'}"
        )
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
