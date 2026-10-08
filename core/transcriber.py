import os
import sys
import shutil
from pydub import AudioSegment
from sarvamai import SarvamAI

# Optional Whisper support for local environments
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    whisper = None
    WHISPER_AVAILABLE = False

# ── Ensure FFmpeg is available for Pydub & Whisper ───────────────────────────
venv_scripts = os.path.dirname(sys.executable)
if venv_scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

# Fallback to imageio_ffmpeg if available
if not shutil.which("ffmpeg"):
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    AudioSegment.converter = ffmpeg_bin
    AudioSegment.ffmpeg = ffmpeg_bin

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

def get_sarvam_api_key():
    return os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_AI_API_KEY")

_sarvam_client = None

def get_sarvam_client() -> SarvamAI:
    global _sarvam_client
    api_key = get_sarvam_api_key()
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY (or SARVAM_AI_API_KEY) is not set in environment / .env")
    if _sarvam_client is None:
        _sarvam_client = SarvamAI(api_subscription_key=api_key)
    return _sarvam_client

_model = None

def load_model():
    global _model  
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "Whisper is not installed. To use speech-to-text, please configure SARVAM_API_KEY."
        )
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    if _model is None: 
        print(f"Loading Whisper model: {whisper_model} ...")
        _model = whisper.load_model(whisper_model) 
        print("Whisper model loaded.")
    return _model 


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()  
    result = model.transcribe(chunk_path, task="transcribe")  
    return result.get("text", "")  


def _send_to_sarvam(piece_path: str, language: str = "hinglish") -> str:
    """Send one ≤30s WAV file to Sarvam via the official SDK and return the English transcript."""
    client = get_sarvam_client()
    sarvam_model = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

    with open(piece_path, "rb") as f:
        file_tuple = (os.path.basename(piece_path), f, "audio/wav")
        # Hinglish route or saaras:v2.5 model translates Indic speech to English
        if language.lower() == "hinglish" or "v2.5" in sarvam_model:
            response = client.speech_to_text.translate(
                file=file_tuple,
                model="saaras:v2.5",
            )
        else:
            # General / English transcription
            response = client.speech_to_text.transcribe(
                file=file_tuple,
                model=sarvam_model,
                language_code="en-IN" if language.lower() == "english" else "unknown",
            )

    return getattr(response, "transcript", "") or ""


def transcribe_chunk_sarvam(chunk_path: str, language: str = "hinglish") -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not get_sarvam_api_key():
        raise RuntimeError("SARVAM_API_KEY (or SARVAM_AI_API_KEY) is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ({language}) ...")
            full_text += _send_to_sarvam(piece_path, language=language) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Sarvam or Whisper depending on language choice and environment.
    - hinglish → Sarvam (translates to English while transcribing)
    - english  → Sarvam AI (cloud-ready) or local Whisper if configured
    """
    lang = language.lower()
    if lang == "hinglish":
        return transcribe_chunk_sarvam(chunk_path, language="hinglish")

    # English route:
    use_local_whisper = os.getenv("USE_LOCAL_WHISPER", "false").lower() in ("true", "1")
    if use_local_whisper and WHISPER_AVAILABLE:
        return transcribe_chunk_whisper(chunk_path)

    # Prefer Sarvam AI for cloud deployment and when SARVAM_API_KEY is available
    if get_sarvam_api_key():
        return transcribe_chunk_sarvam(chunk_path, language="english")

    # Fallback to local Whisper if installed
    if WHISPER_AVAILABLE:
        return transcribe_chunk_whisper(chunk_path)

    raise RuntimeError(
        "Neither Sarvam AI API key nor local Whisper is available. "
        "Please provide SARVAM_API_KEY in your environment / Streamlit secrets."
    )


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    lang = language.lower()

    if lang == "hinglish":
        engine = "Sarvam AI (Hinglish → English)"
    elif os.getenv("USE_LOCAL_WHISPER", "false").lower() in ("true", "1") and WHISPER_AVAILABLE:
        engine = "Whisper (Local)"
    elif get_sarvam_api_key():
        engine = "Sarvam AI (English STT)"
    elif WHISPER_AVAILABLE:
        engine = "Whisper (Local)"
    else:
        engine = "Sarvam AI"

    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()
