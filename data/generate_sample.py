import numpy as np
import soundfile as sf

# Generate a short synthetic audio (2 seconds at 16kHz)
# This is just for testing - replace with real Hindi audio
sample_rate = 16000
duration = 2  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
# Simple sine wave as placeholder
waveform = np.sin(2 * np.pi * 440 * t)

# Save to file
sf.write("data/sample_hindi.wav", waveform, sample_rate)
print("Sample audio file created: data/sample_hindi.wav")
print("Note: This is synthetic audio. Replace with actual Hindi speech for real inference.")
