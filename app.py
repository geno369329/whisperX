from flask import Flask, request, jsonify
import torch
import whisperx
import os
import tempfile
import requests

app = Flask(__name__)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.json
    file_url = data.get("url")
    notion_page_id = data.get("notionPageId")
    video_format = data.get("format")  # e.g. "Shortform" or "Longform"
    webhook_url = data.get("webhookUrl")

    # ✅ Webhook fallback logic (only change)
    if not webhook_url:
        webhook_url = "https://ehmokeh.app.n8n.cloud/webhook-test/e33cf31c-a80d-4115-98e9-160f2103f0c7"
        if "RAILWAY_ENVIRONMENT" in os.environ and "prod" in os.environ["RAILWAY_ENVIRONMENT"].lower():
            webhook_url = "https://ehmokeh.app.n8n.cloud/webhook/e33cf31c-a80d-4115-98e9-160f2103f0c7"

    if not file_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Download video file to temp location
        response = requests.get(file_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download file"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            audio_path = tmp.name

        # Load model with float32 compute type for better CPU compatibility
        model = whisperx.load_model("large-v3", device, compute_type="float32")

        # Transcribe
        result = model.transcribe(audio_path)

        # Align for accurate word-level timestamps
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)

        os.remove(audio_path)

        response_payload = {
            "notionPageId": notion_page_id,
            "format": video_format,
            "language": result["language"],
            "segments": result_aligned["segments"]
        }

        # Send to webhook
        if webhook_url:
            requests.post(webhook_url, json=response_payload)

        return jsonify({"status": "Transcription complete", **response_payload})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
