# CUA WorkTalk

A real-time, voice-to-voice English–Spanish translator built for The Catholic University of America's facilities staff. A worker taps a button and speaks, and the app transcribes the speech, translates it with workplace-specific vocabulary, and plays the translation aloud. Two colleagues can join the same "room" code on their phones so each hears the other's translated messages.

Built as a course project for Natural Language Processing (AI 546) at The Catholic University of America.

## How it works

```
Speech (browser mic) → Whisper (speech-to-text) → GPT-4o-mini translation
                     + workplace vocabulary hints → TTS (text-to-speech) → audio playback
```

1. **Record:** the browser captures audio and stops automatically after about 1.2 seconds of silence.
2. **Transcribe:** the audio is sent to OpenAI Whisper (`whisper-1`).
3. **Translate:** GPT-4o-mini translates English ↔ Spanish. Before translating, the app looks up any matching terms in `vocab.json` (campus locations, cleaning equipment, common workplace phrases) and adds them to the prompt so those terms are translated consistently. This is a simple retrieval-augmented approach.
4. **Speak:** the translation is converted to audio with OpenAI TTS (`tts-1`) and played back.
5. **Share:** if both users join the same room code, the latest translated message and audio are delivered to the other person's device.

## Features

- Hands-free recording with automatic silence detection
- English → Spanish and Spanish → English modes, with the interface shown in the selected language
- Domain vocabulary (`vocab.json`) for locations, equipment, and safety phrases used by facilities teams
- Room codes so two people can communicate across phones
- Mobile-friendly single-page interface

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn
- **AI models:** OpenAI Whisper (speech-to-text), GPT-4o-mini (translation), TTS-1 (speech synthesis)
- **Frontend:** HTML, CSS, JavaScript (MediaRecorder API)
- **Deployment:** Docker, Render

## Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: transcription, translation, speech synthesis, and room endpoints |
| `index.html` | Single-page web interface |
| `vocab.json` | English–Spanish workplace vocabulary used to guide translations |
| `test_api.py` | Script for testing translations from the command line |
| `Dockerfile` | Container setup for deployment |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
git clone https://github.com/Fancyk-hash/worktalk.git
cd worktalk
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
uvicorn main:app --reload
```

Then open http://localhost:8000 in a browser and allow microphone access.

Or with Docker:

```bash
docker build -t worktalk .
docker run -p 8000:8000 -e OPENAI_API_KEY="your-key-here" worktalk
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `POST` | `/translate` | Takes an audio file, language (`en` or `es`), and optional room code; returns the original text, translation, and audio URL |
| `GET` | `/audio` | Returns the latest translated audio |
| `GET` | `/room/{room_code}/latest` | Returns a new message for a room, if any |
| `GET` | `/room/{room_code}/audio` | Returns the latest audio for a room |

## Limitations and next steps

- Room messages and audio are stored as local files, so the app is meant for small demos rather than many simultaneous users.
- Supports English and Spanish only.
- Possible improvements: a database for rooms, more languages, and an evaluation of translation quality with and without the vocabulary hints.

## Author

Fancy Kiruok, M.S. Data Analytics, The Catholic University of America
