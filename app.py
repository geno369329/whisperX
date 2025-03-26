from flask import Flask, request, jsonify
import torch
import whisperx
import os
import tempfile
import requests
import threading

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.json
    file_url = data.get("url")
    notion_page_id = data.get("notionPageId")
    video_format = data.get("format")
    webhook_url = data.get("webhookUrl")

    if not webhook_url:
        webhook_url = "https://ehmokeh.app.n8n.cloud/webhook-test/e33cf31c-a80d-4115-98e9-160f2103f0c7"
        if "RAILWAY_ENVIRONMENT" in os.environ and "prod" in os.environ["RAILWAY_ENVIRONMENT"].lower():
            webhook_url = "https://ehmokeh.app.n8n.cloud/webhook/e33cf31c-a80d-4115-98e9-160f2103f0c7"

    if not file_url:
        return jsonify({"error": "No URL provided"}), 400

    def process_transcription():
        try:
            print("🟡 Starting transcription thread...")

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

            os.remove(audio_path)

            response_payload = {
                "notionPageId": notion_page_id,
                "format": video_format,
                "language": result["language"],
                "segments": result_aligned["segments"]
            }

            print("📬 Sending transcription to webhook:", webhook_url)
            res = requests.post(webhook_url, json=response_payload)
            print("✅ Webhook response status:", res.status_code)

        except Exception as e:
            print("❌ Transcription error:", str(e))

    threading.Thread(target=process_transcription).start()
    return jsonify({"status": "Accepted"}), 202

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
