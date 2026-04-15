from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import json, os, tempfile, time
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

with open("vocab.json", "r") as f:
    vocab = json.load(f)

rooms = {}

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
async def translate_audio(
    audio: UploadFile = File(...),
    lang: str = Form("en"),
    room: str = Form(None)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    original_text = transcription if isinstance(transcription, str) else transcription.text
    os.unlink(tmp_path)

    if lang == "en":
        from_lang, to_lang = "English", "Spanish"
    else:
        from_lang, to_lang = "Spanish", "English"

    translated_text = translate(original_text, from_lang, to_lang)

    tts_response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=translated_text
    )

    # Save global audio
    tts_response.stream_to_file("response.mp3")

    # Save room-specific audio if room provided
    if room:
        room = room.upper()
        import shutil
        shutil.copy("response.mp3", f"room_{room}.mp3")
        rooms[room] = {
            "original": original_text,
            "translated": translated_text,
            "timestamp": time.time(),
            "seen": False
        }

    return {
        "original": original_text,
        "translated": translated_text,
        "audio_url": "/audio"
    }

@app.get("/audio")
async def get_audio():
    return FileResponse("response.mp3", media_type="audio/mpeg")

@app.get("/room/{room_code}/latest")
async def get_room_latest(room_code: str):
    room = rooms.get(room_code.upper())
    if not room or room.get("seen"):
        return {"has_new": False}
    room["seen"] = True
    return {
        "has_new": True,
        "original": room["original"],
        "translated": room["translated"]
    }

@app.get("/room/{room_code}/audio")
async def get_room_audio(room_code: str):
    path = f"room_{room_code.upper()}.mp3"
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")
    return FileResponse("response.mp3", media_type="audio/mpeg")
