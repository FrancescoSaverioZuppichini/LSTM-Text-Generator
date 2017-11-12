from flask import Flask
from flask import jsonify

app = Flask(__name__)

from generate import generate_from

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/api/generate/<text>/<int:n_text>')
def generate_text(text, n_text):

    return jsonify(result=generate_from('../checkpoints/shakespeare',text,n_text))

app.run(port=3030, debug=True)