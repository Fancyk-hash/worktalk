from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import json, os, tempfile, time, asyncio
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Room storage — room_code -> list of connected websockets
rooms = {}
room_data = {}

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

@app.websocket("/ws/{room_code}")
async def websocket_endpoint(websocket: WebSocket, room_code: str):
    await websocket.accept()
    room_code = room_code.upper()
    if room_code not in rooms:
        rooms[room_code] = []
    rooms[room_code].append(websocket)
    count = len(rooms[room_code])
    await websocket.send_json({"type": "joined", "count": count})
    for ws in rooms[room_code]:
        try:
            await ws.send_json({"type": "count", "count": count})
        except:
            pass
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        rooms[room_code].remove(websocket)
        for ws in rooms[room_code]:
            try:
                await ws.send_json({"type": "count", "count": len(rooms[room_code])})
            except:
                pass

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
        input=translated_text,
        speed=1.0
    )

    audio_path = "response.mp3"
    tts_response.stream_to_file(audio_path)

    # Notify all OTHER devices in the room via WebSocket
    if room:
        room = room.upper()
        if room in rooms:
            for ws in rooms[room]:
                try:
                    await ws.send_json({
                        "type": "translation",
                        "original": original_text,
                        "translated": translated_text,
                        "audio_url": "/audio?t=" + str(time.time())
                    })
                except:
                    pass

    return {
        "original": original_text,
        "translated": translated_text,
        "audio_url": "/audio"
    }

@app.get("/audio")
async def get_audio():
    return FileResponse("response.mp3", media_type="audio/mpeg")
