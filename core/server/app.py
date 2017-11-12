from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/generate/<text>')
def generate_text(text):
    return text