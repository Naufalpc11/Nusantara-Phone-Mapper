from flask import Flask, request, jsonify
from flask_cors import CORS

from services.phoneme_mapper import map_word

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return {"message": "Backend jalan 🔥"}


@app.route("/api/map", methods=["POST"])
def map_api():
    data = request.json

    source = data.get("source")
    targets = data.get("targets")

    result = map_word(source, targets)

    return jsonify({
        "source": source,
        "targets": targets,
        "result": result
    })


if __name__ == "__main__":
    app.run(debug=True)