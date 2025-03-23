from flask import Flask, request, jsonify
import whisperx
import os
import tempfile

app = Flask(__name__)

# Load model once at startup
device = "cuda" if whisperx.utils.get_device() == "cuda" else "cpu"
model = whisperx.load_model("large-v2", device)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.save(tmp.name)
        result = model.transcribe(tmp.name)
        os.remove(tmp.name)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
