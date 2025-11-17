from flask import Flask, request, render_template
from model.inference import generate_subject

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    subject = ""
    if request.method == 'POST':
        body = request.form.get('body', '').strip()
        if body:
            subject = generate_subject(body)
    return render_template('index.html', subject=subject)

if __name__ == '__main__':
    app.run(debug=True)