from flask import Flask, jsonify, request
import json
from datetime import datetime

app = Flask(__name__)

# Path to your saved JSON files
TOKEN_JSON_FILE = 'mockserver/token.json'
AZLOGS_JSON_FILE = 'mockserver/AzDiag.json'


@app.route('/token', methods=['POST'])
def mock_token():
    with open(TOKEN_JSON_FILE, 'r') as file:
        data = json.load(file)
    # Calculate expires_in based on expires_at
    # expires_at = datetime.fromisoformat(data['expires_on'])
    # now = datetime.now()
    # expires_in_seconds = int((expires_at - now).total_seconds())
    # data['expires_in'] = expires_in_seconds

    return jsonify(data)


@app.route('/query', methods=['POST'])
def mock_azlogs():
    with open(AZLOGS_JSON_FILE, 'r') as file:
        data = json.load(file)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
