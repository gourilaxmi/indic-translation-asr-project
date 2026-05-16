from gtts import gTTS
import os

os.makedirs("sample_inputs", exist_ok=True)

samples = [
    ("வணக்கம், என் பெயர் ராஜேஷ். நான் சென்னையில் வசிக்கிறேன்.", "sample1.mp3"),
    ("தமிழ் மொழி மிகவும் அழகான மொழி. இது உலகின் பழமையான மொழிகளில் ஒன்று.", "sample2.mp3"),
    ("இன்று வானிலை மிகவும் நன்றாக இருக்கிறது. வெளியே செல்லலாம்.", "sample3.mp3"),
    ("செயற்கை நுண்ணறிவு தொழில்நுட்பம் இன்று மிகவும் வேகமாக வளர்ந்து வருகிறது.", "sample4.mp3"),
    ("தமிழ்நாட்டில் பல அழகான கோவில்கள் உள்ளன. அவை நம் கலாச்சாரத்தின் சின்னங்கள்.", "sample5.mp3"),
]

for text, filename in samples:
    tts = gTTS(text, lang='ta')
    tts.save(f"sample_inputs/{filename}")
    print(f"Saved: {filename}")

