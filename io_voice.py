import os, json, pyttsx3, pyaudio
from vosk import Model, KaldiRecognizer

_engine = pyttsx3.init()
_engine.setProperty("rate", 185)

def speak_tts(text):
    if not text.strip():
        return
    _engine.say(text)
    _engine.runAndWait()

class VoiceListener:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found: {model_path}")
        print("🎧 Using Vosk model:", model_path)
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, 16000)
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                                   input=True, frames_per_buffer=8000)
        self.stream.start_stream()

    def listen_once(self):
        print("🎤 Speak now...")
        while True:
            data = self.stream.read(4000, exception_on_overflow=False)
            if self.rec.AcceptWaveform(data):
                text = json.loads(self.rec.Result()).get("text", "")
                if text:
                    print("🗣 You:", text)
                    return text
