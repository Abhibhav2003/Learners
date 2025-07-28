# register_voice.py
import sounddevice as sd
from scipy.io.wavfile import write
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

def record_audio(filename, duration=4, fs=16000):
    print("Recording registration voice... Please speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"Audio recorded and saved as {filename}")

def register_student(student_id):
    filename = f"{student_id}_registered.wav"
    record_audio(filename)

    encoder = VoiceEncoder()
    wav = preprocess_wav(filename)
    embedding = encoder.embed_utterance(wav)

    np.save(f"{student_id}_embedding.npy", embedding)
    print(f"Voice embedding saved as {student_id}_embedding.npy")

# ---- Run this script ----
if __name__ == "__main__":
    student_id = input("Enter student ID: ")
    register_student(student_id)