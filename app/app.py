import os
import requests
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = os.path.join(os.getcwd(), 'static')

logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        logging.debug(f"Current working directory: {os.getcwd()}")
        
        image = request.files["file"]
        if image.filename == '':
            logging.debug("Filename is invalid")
            return redirect(request.url)

        filename = secure_filename(image.filename)
        logging.debug(f"Filename received: {filename}")

        basedir = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename)
        image.save(img_path)
        logging.debug(f"Image saved to: {img_path}")

        try:
            res = requests.post("http://torchserve-mar:8080/predictions/mnist", files={'data': open(img_path, 'rb')})
            logging.debug(f"Status code: {res.status_code}")
            prediction = res.text
            logging.debug(f"Response content: {prediction}")
        
        except Exception as e:
            logging.error(f"Exception occurred: {str(e)}")
            prediction = f"Error: {str(e)}"

        return render_template('index.html', prediction_text=f'Prediction: {prediction}')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
