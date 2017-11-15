from flask import Flask
from flask import jsonify
import os

app = Flask(__name__)

from model.generate import generate_from

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/generate/<text>/<int:n_text>')
def generate_text(text, n_text):

    text = generate_from(os.path.abspath('./checkpoints/shakespeare'), text, n_text)

    return jsonify(result=text)

if __name__ == "__main__":
    app.run()

# if __name__ == '__main__':
#     app.run(port=3030, debug=True)