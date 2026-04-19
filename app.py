from flask import Flask, request, jsonify
from flask_cors import CORS
from services.phoneme_mapper import find_closest

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"message": "Backend jalan 🔥"}

# 👇 INI YANG NAMANYA /api/map
@app.route("/api/map", methods=["POST"])
def map_phoneme():
    data = request.json

    source = data.get("source")
    targets = data.get("targets")

    result = find_closest(source, targets)

    return jsonify({
        "source": source,
        "targets": targets,
        "result": result
    })
    


if __name__ == "__main__":
    app.run(debug=True)