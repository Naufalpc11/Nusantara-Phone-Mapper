from flask import Flask, request, jsonify
from flask_cors import CORS

from services.phoneme_mapper import map_word

# Inisialisasi backend API sederhana.
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    """Endpoint health check untuk memastikan backend aktif."""
    return {"message": "Backend Nusantara Phone Mapper aktif"}


@app.route("/api/map", methods=["POST"])
def map_api():
    """Terima source+targets lalu kembalikan hasil mapping terbaik."""
    # Ambil payload dari request frontend/client.
    data = request.json

    source = data.get("source")
    targets = data.get("targets")

    # Jalankan mapping satu kata terhadap daftar kandidat target.
    result = map_word(source, targets)

    return jsonify({
        "source": source,
        "targets": targets,
        "result": result
    })


if __name__ == "__main__":
    app.run(debug=True)