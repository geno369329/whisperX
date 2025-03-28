from flask import Flask, request, jsonify
import torch
import whisperx
import os
import tempfile
import requests
from rq import Queue
from redis import Redis
from dotenv import load_dotenv
# from pyannote.audio import Pipeline  # ⛔️ Commented out for now
import torchaudio
import time

load_dotenv()

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
redis_conn = Redis.from_url(os.getenv("REDIS_URL"))
q = Queue(connection=redis_conn)


def estimate_transcription_time(duration_sec):
    if device == "cuda":
        return round(duration_sec * 0.5)
    else:
        return round(duration_sec * 2.5)


def get_audio_duration(file_path):
    metadata = torchaudio.info(file_path)
    return metadata.num_frames / metadata.sample_rate


def process_transcription(file_url, notion_page_id, video_format, final_webhook):
    try:
        print("🟡 Starting transcription job...")

        response = requests.get(file_url, stream=True)
        if response.status_code != 200:
            print("❌ Failed to download video. Status code:", response.status_code)
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            audio_path = tmp.name

        print("📥 Download complete. Loading WhisperX model...")
        model = whisperx.load_model("large-v3", device, compute_type="float32")
        result = model.transcribe(audio_path)

        print("🧠 Transcription complete. Aligning...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)

        # 🛑 Speaker diarization disabled temporarily
        # print("🧑‍🤝‍🧑 Running speaker diarization...")
        # diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_TOKEN"))
        # diarize_segments = diarize_pipeline(audio_path)
        # result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result_aligned["word_segments"])

        os.remove(audio_path)

        response_payload = {
            "notionPageId": notion_page_id,
            "format": video_format,
            "language": result["language"],
            "segments": result_aligned["segments"],
            # "words": result_with_speakers  # ⛔️ Also temporarily removed
        }

        print("📬 Sending transcription to webhook:", final_webhook)
        res = requests.post(final_webhook, json=response_payload)
        print("✅ Webhook response status:", res.status_code)

    except Exception as e:
        print("❌ Transcription error:", str(e))


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.json
    file_url = data.get("url")
    notion_page_id = data.get("notionPageId")
    video_format = data.get("format")
    incoming_webhook = data.get("webhookUrl")

    if incoming_webhook == "https://ehmokeh.app.n8n.cloud/webhook-test/ba47d62c-3247-43e2-a834-906dffb943dd":
        final_webhook = "https://ehmokeh.app.n8n.cloud/webhook-test/e33cf31c-a80d-4115-98e9-160f2103f0c7"
    elif incoming_webhook == "https://ehmokeh.app.n8n.cloud/webhook/ba47d62c-3247-43e2-a834-906dffb943dd":
        final_webhook = "https://ehmokeh.app.n8n.cloud/webhook/e33cf31c-a80d-4115-98e9-160f2103f0c7"
    else:
        return jsonify({"error": "Unrecognized webhook URL"}), 400

    if not file_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        print("📏 Estimating transcription time...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with requests.get(file_url, stream=True) as r:
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
        tmp.close()
        duration = get_audio_duration(tmp.name)
        eta_sec = estimate_transcription_time(duration)
        os.remove(tmp.name)
    except Exception as e:
        print("⚠️ Failed to estimate duration:", str(e))
        eta_sec = None

    job = q.enqueue(process_transcription, file_url, notion_page_id, video_format, final_webhook)
    print(f"📦 Enqueued job ID: {job.id}")

    return jsonify({
        "status": "Accepted",
        "jobId": job.id,
        "estimatedTimeSec": eta_sec
    }), 202
