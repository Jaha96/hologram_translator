from flask import Flask, url_for
from flask import render_template
from markupsafe import escape
from OpenNMT.onmt.bin.translate import main

app = Flask(__name__)
# url_for('static/css/', filename='style.css')

@app.route('/')
def hello_world(text=None):
    text="こんにちは！"
    return render_template("hologram.html", text=text)

@app.route('/hello')
def hello():
    return 'This is from hello'

@app.route('/translate/<text>')
def profile(text):
    jp_text = main(text)
    print("jp_text", jp_text)
    return render_template("hologram.html", text=jp_text)

@app.errorhandler(404)
def not_found(error):
    return render_template("hologram.html", text="Not found")
