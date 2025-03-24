from flask import Flask, request, jsonify
import whisperx

app = Flask(__name__)

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files["file"]
    audio_path = f"/tmp/{audio_file.filename}"
    audio_file.save(audio_path)

    model = whisperx.load_model("large-v2", device="cuda" if whisperx.utils.get_device() == "cuda" else "cpu")
    result = model.transcribe(audio_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
