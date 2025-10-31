from pathlib import Path
from io_voice import VoiceListener, speak_tts
from nlp import GPTClient
from agents import Orchestrator, Ctx
from speech_corrector import SpeechCorrector
from memory_manager import MemoryManager

def main():
    model_dir = Path(__file__).resolve().parent / "models" / "vosk-model-small-en-us-0.15"
    gpt = GPTClient(model="gpt-4o-mini")
    corrector = SpeechCorrector(gpt)
    memory_mgr = MemoryManager(gpt)
    orch = Orchestrator(gpt)
    ctx = Ctx()

    try:
        listener = VoiceListener(str(model_dir))
    except Exception as e:
        print(f"[Audio Error] {e}")
        print("Voice input unavailable — switching to text mode.")
        listener = None

    speak_tts("RescueHub is listening. Please describe your emergency.")
    print("=== RescueHub Started ===")

    while True:
        if ctx.done:
            print("Conversation complete — exiting gracefully.")
            break

        if listener:
            user_raw = listener.listen_once()
        else:
            user_raw = input("Type your message: ")

        if not user_raw.strip():
            continue

        user_text = corrector.correct(user_raw, orch.memory.get_summary())
        print(f"LLM Text-Corrected: {user_text}")

        memory_context = memory_mgr.recall_context(user_text)
        if memory_context:
            print(f"Memory recall: {memory_context}")
            user_text = f"(Context: {memory_context})\n{user_text}"

        reply, ctx = orch.step(user_text, ctx)
        print(reply)

        memory_mgr.add_entry(user_text, reply)

        try:
            speak_tts(reply.split(": ", 1)[1])
        except Exception as e:
            print(f"[TTS Error] {e}")

    speak_tts("Help is on the way. Stay safe.")

if __name__ == "__main__":
    main()
