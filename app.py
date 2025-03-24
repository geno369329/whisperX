from flask import Flask, request, jsonify
import torch
import whisperx
import os
import tempfile

app = Flask(__name__)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        file.save(tmp.name)
        audio_path = tmp.name

    try:
        # Load model
        model = whisperx.load_model("large-v3", device)

        # Transcribe
        result = model.transcribe(audio_path)

        # Align (optional but improves word-level timestamps)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)

        # Clean up temp file
        os.remove(audio_path)

        return jsonify({
            "segments": result_aligned["segments"],
            "language": result["language"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
