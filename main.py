from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import os
import tempfile
from openai import OpenAI

app = FastAPI()

import os
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load workplace vocabulary
with open("vocab.json", "r") as f:
    vocab = json.load(f)

def get_relevant_vocab(text):
    relevant = {}
    text_lower = text.lower()
    for category, words in vocab.items():
        for english, spanish in words.items():
            if english.lower() in text_lower or spanish.lower() in text_lower:
                relevant[english] = spanish
    return relevant

def translate(text, from_lang, to_lang):
    relevant_vocab = get_relevant_vocab(text)
    vocab_hint = ""
    if relevant_vocab:
        vocab_hint = "\nUse these exact translations for workplace terms:\n"
        for eng, esp in relevant_vocab.items():
            vocab_hint += f"- {eng} = {esp}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a workplace translator for CUA facilities department.
Translate from {from_lang} to {to_lang}.
Be natural and concise.{vocab_hint}"""},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.post("/translate")
async def translate_audio(audio: UploadFile = File(...), lang: str = "en"):
    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Step 1: Speech to text with Whisper
    with open(tmp_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    
    original_text = transcription.text
    os.unlink(tmp_path)

    # Step 2: Detect language and translate
    if lang == "en":
        from_lang, to_lang = "English", "Spanish"
    else:
        from_lang, to_lang = "Spanish", "English"

    translated_text = translate(original_text, from_lang, to_lang)

    # Step 3: Text to speech
    tts_response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=translated_text
    )

    # Save audio response
    audio_path = "response.mp3"
    tts_response.stream_to_file(audio_path)

    return {
        "original": original_text,
        "translated": translated_text,
        "audio_url": "/audio"
    }

@app.get("/audio")
async def get_audio():
    return FileResponse("response.mp3", media_type="audio/mpeg")
