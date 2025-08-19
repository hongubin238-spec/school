    # main.py
    import os, base64, tempfile, pathlib
    from fastapi import FastAPI, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from openai import OpenAI
    import requests

    app = FastAPI(title="School Voice Bot MVP")

    # Allow CORS for testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    # Serve static files (client.html later)
    STATIC_DIR = pathlib.Path("static")
    STATIC_DIR.mkdir(exist_ok=True, parents=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # OpenAI client
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY in Replit Secrets")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # --- STT ---
    def stt_whisper_ko(file_path: str) -> str:
        with open(file_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ko"
            )
        return resp.text.strip()

    # --- GPT ---
    def answer_with_gpt(question: str) -> str:
        system = "You are a helpful Korean school assistant. Answer concisely in Korean."
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return chat.choices[0].message.content.strip()

    # --- TTS ---
    def tts_openai_ko_mp3(text: str) -> bytes:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "gpt-4o-mini-tts", "voice": "alloy", "input": text, "format": "mp3"}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.content

    @app.get("/")
    def root():
        return HTMLResponse("<h3>Server running. Open /static/client.html later.</h3>")

    @app.post("/api/ask-audio")
    async def ask_audio(audio: UploadFile = File(...)):
        suffix = pathlib.Path(audio.filename or "audio").suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            transcript = stt_whisper_ko(tmp_path)
            answer = answer_with_gpt(transcript)
            mp3_bytes = tts_openai_ko_mp3(answer)
            audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
            return JSONResponse({
                "transcript": transcript,
                "answer": answer,
                "audio_b64": audio_b64,
                "mime": "audio/mpeg"
            })
        finally:
            try: os.remove(tmp_path)
            except: pass
