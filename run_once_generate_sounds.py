# run_once_generate_sounds.py
import wave
import struct
import math
import os

os.makedirs('sounds', exist_ok=True)

def make_beep(filename, frequency=440, duration=0.4, volume=0.5):
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        
        for i in range(num_samples):
            # Generate sine wave without numpy
            value = math.sin(2 * math.pi * frequency * i / sample_rate)
            packed = struct.pack('<h', int(value * volume * 32767))
            f.writeframes(packed)
    
    print(f"Created: {filename}")

make_beep('sounds/alert_mild.wav',     frequency=440,  duration=0.3)
make_beep('sounds/alert_warning.wav',  frequency=700,  duration=0.5)
make_beep('sounds/alert_critical.wav', frequency=1000, duration=0.8)

print("All 3 sound files created!")