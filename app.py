from flask import Flask, request, jsonify
import torch
import whisperx
import os
import tempfile
import requests
from rq import Queue
from redis import Redis
from dotenv import load_dotenv
from pyannote.audio import Pipeline  # ✅ NEW

load_dotenv()

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
redis_conn = Redis.from_url(os.getenv("REDIS_URL"))
q = Queue(connection=redis_conn)


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

        print("🧑‍🤝‍🧑 Running speaker diarization...")  # ✅ NEW
        diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_TOKEN"))
        diarize_segments = diarize_pipeline(audio_path)
        result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result_aligned["word_segments"])

        os.remove(audio_path)

        response_payload = {
            "notionPageId": notion_page_id,
            "format": video_format,
            "language": result["language"],
            "segments": result_aligned["segments"],
            "words": result_with_speakers  # ✅ NEW: word-level output with speaker tags
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

    # 🔁 Determine response webhook based on source
    if incoming_webhook == "https://ehmokeh.app.n8n.cloud/webhook-test/ba47d62c-3247-43e2-a834-906dffb943dd":
        final_webhook = "https://ehmokeh.app.n8n.cloud/webhook-test/e33cf31c-a80d-4115-98e9-160f2103f0c7"
    elif incoming_webhook == "https://ehmokeh.app.n8n.cloud/webhook/ba47d62c-3247-43e2-a834-906dffb943dd":
        final_webhook = "https://ehmokeh.app.n8n.cloud/webhook/e33cf31c-a80d-4115-98e9-160f2103f0c7"
    else:
        return jsonify({"error": "Unrecognized webhook URL"}), 400

    if not file_url:
        return jsonify({"error": "No URL provided"}), 400

    # 👇 Enqueue the transcription job
    job = q.enqueue(process_transcription, file_url, notion_page_id, video_format, final_webhook)
    print(f"📦 Enqueued job ID: {job.id}")

    return jsonify({"status": "Accepted", "jobId": job.id}), 202
