# speech_recog.py
import sounddevice as sd
from scipy.io.wavfile import write
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import random
from numpy.linalg import norm

def record_audio(filename, duration=4, fs=16000):
    print("Recording login voice... Please repeat the phrase displayed.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"Audio recorded and saved as {filename}")

def get_random_phrase():
    phrases = [
        "Artificial intelligence is the future.",
        "The quick brown fox jumps over the lazy dog.",
        "Today is a beautiful day.",
        "I love learning new things.",
        "Technology is evolving rapidly."
    ]
    return random.choice(phrases)

def verify_student(student_id):
    phrase = get_random_phrase()
    print(f"\nPlease say this phrase clearly:\n\"{phrase}\"\n")

    login_filename = f"{student_id}_login.wav"
    record_audio(login_filename)

    encoder = VoiceEncoder()
    login_wav = preprocess_wav(login_filename)
    login_embedding = encoder.embed_utterance(login_wav)

    try:
        stored_embedding = np.load(f"{student_id}_embedding.npy")
    except FileNotFoundError:
        print("Error: No registered voice found. Please register first.")
        return

    # Cosine similarity
    similarity = np.dot(stored_embedding, login_embedding) / (norm(stored_embedding) * norm(login_embedding))
    print(f"Similarity Score: {similarity:.2f}")

    # Threshold check
    if similarity > 0.75:
        print("Voice Verified: Login Successful.")
    else:
        print("Voice Mismatch: Login Denied.")

# ---- Run this script ----
if __name__ == "__main__":
    student_id = input("Enter student ID: ")
    verify_student(student_id)