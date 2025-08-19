# main.py
import os, json, base64, tempfile, pathlib, datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI
import requests
import pandas as pd
import pytz

# -------------------- App --------------------
app = FastAPI(title="School Voice Bot MVP")

# Allow all for MVP demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Static mount --------------------
STATIC_DIR = pathlib.Path("static")
STATIC_DIR.mkdir(exist_ok=True, parents=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -------------------- Data storage --------------------
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
DATA_FILE = DATA_DIR / "school_data.json"

# -------------------- OpenAI --------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in Replit Secrets")

client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------- Helpers --------------------
def stt_whisper_ko(file_path: str) -> str:
    """Transcribe Korean speech to text via Whisper API."""
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ko",
        )
    return (resp.text or "").strip()


def build_system_prompt() -> str:
    """System prompt with today's date (KST) included."""
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    today_str = now.strftime("%Y-%m-%d")
    weekday_ko = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]

    return (
        f"당신은 한국 고등학교 학생들을 돕는 교내 AI 비서입니다.\n"
        f"오늘은 {today_str} ({weekday_ko}요일)입니다.\n"
        "학생이 '오늘', '내일', '이번 주' 같은 날짜 표현을 쓰면 반드시 한국 시간 기준으로 계산하세요.\n"
        "JSON 데이터에 급식표, 시간표, 학사일정이 포함되어 있으면 그 정보를 참고하세요.\n"
        "데이터에 없는 경우는 '정보가 없습니다'라고 간단히 말하세요.\n"
        "답변은 한국어로 간결하게 해주세요."
    )


def answer_with_gpt(question: str, data_hint: Optional[str] = None) -> str:
    """Ask GPT with optional JSON slice."""
    user_content = question if not data_hint else f"""질문: {question}

(참고 데이터 일부; JSON)
{data_hint}"""
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": build_system_prompt()
            },
            {
                "role": "user",
                "content": user_content
            },
        ],
    )
    return chat.choices[0].message.content.strip()


def tts_openai_ko_mp3(text: str) -> bytes:
    """Synthesize Korean speech (MP3) via OpenAI TTS."""
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "tts-1",
        "voice": "alloy",
        "input": text,
        "format": "mp3"
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.content  # MP3 bytes


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Try to normalize likely date columns to ISO (YYYY-MM-DD)."""
    out = df.copy()
    for col in out.columns:
        cs = str(col).lower()
        if "date" in cs or "날짜" in cs or "일자" in cs:
            try:
                out[col] = pd.to_datetime(out[col],
                                          errors="coerce").dt.date.astype(str)
            except Exception:
                pass
    return out


def load_data_snippet(max_chars: int = 12000) -> str:
    """Return the first part of saved JSON to keep tokens small."""
    if not DATA_FILE.exists():
        return ""
    text = DATA_FILE.read_text(encoding="utf-8")
    return text[:max_chars]


# -------------------- Basic routes --------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return HTMLResponse(
        "<html><body>"
        "<h3>School Voice Bot MVP</h3>"
        "<p>Teacher: open <code>/teacher</code> to upload Excel.</p>"
        "<p>Student: open <code>/client</code> to ask by voice/text.</p>"
        "</body></html>")


@app.get("/client")
def get_client():
    return FileResponse("static/client.html", media_type="text/html")


@app.get("/teacher")
def get_teacher():
    return FileResponse("static/teacher.html", media_type="text/html")


# -------------------- Teacher: Excel upload --------------------
@app.post("/api/upload-excel")
async def upload_excel(file: UploadFile = File(...),
                       mode: str = Form("replace")):
    """
    Teacher uploads Excel. We convert all sheets to JSON and save.
    mode: "replace" (overwrite) or "merge" (append rows per sheet name)
    """
    suffix = pathlib.Path(file.filename or "data.xlsx").suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        xls = pd.ExcelFile(tmp_path)
        sheets = {}
        for name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=name)
            df = normalize_dates(df)
            sheets[name] = df.fillna("").to_dict(orient="records")

        payload = {
            "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
            "sheets": sheets,
        }

        if mode == "merge" and DATA_FILE.exists():
            old = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            old_sheets = old.get("sheets", {})
            for k, v in sheets.items():
                if isinstance(old_sheets.get(k), list):
                    old_sheets[k].extend(v)
                else:
                    old_sheets[k] = v
            payload["sheets"] = old_sheets

        DATA_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return {
            "ok": True,
            "sheets": list(payload["sheets"].keys()),
            "mode": mode
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# -------------------- Student: Text QA --------------------
@app.post("/api/ask-text")
async def ask_text(q: str = Form(...)):
    data_hint = load_data_snippet()
    answer = answer_with_gpt(q, data_hint if data_hint else None)
    mp3_bytes = tts_openai_ko_mp3(answer)
    audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
    return {"answer": answer, "audio_b64": audio_b64, "mime": "audio/mpeg"}


# -------------------- Student: Audio QA --------------------
@app.post("/api/ask-audio")
async def ask_audio(audio: UploadFile = File(...)):
    """
    1) Receive audio (webm/mp3/wav)
    2) STT -> transcript
    3) GPT (with JSON hint) -> answer
    4) TTS -> mp3 (base64)
    """
    suffix = pathlib.Path(audio.filename or "audio").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        transcript = stt_whisper_ko(tmp_path)
        data_hint = load_data_snippet()
        answer = answer_with_gpt(transcript, data_hint if data_hint else None)
        mp3_bytes = tts_openai_ko_mp3(answer)
        audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
        return {
            "transcript": transcript,
            "answer": answer,
            "audio_b64": audio_b64,
            "mime": "audio/mpeg",
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
