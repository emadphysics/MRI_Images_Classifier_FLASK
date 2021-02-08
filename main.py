from flask import Flask, redirect, request, jsonify
from keras import models
import numpy as np
from PIL import Image
import io


app = Flask(__name__)
model = None


def load_model():
    global model
    model = models.load_model('model.h5')
    model.summary()
    print('Loaded the model')


@app.route('/')
def index():
    return redirect('/static/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('test.jpg')
        x = np.array(img.resize((150, 150)))
        x = x.reshape(1, 150, 150, 3)

        answ = model.predict_on_batch(x)
        classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

        def names(number):
            if (number == 0):
                return classes[0]
            elif (number == 1):
                return classes[1]
            elif (number == 2):
                return classes[2]
            elif (number == 3):
                return classes[3]

        classification = np.where(answ == np.amax(answ))[1][0]


        confidence = str(answ[0][classification] * 100)
        pred = names(classification)

        data = dict(pred=pred, confidence=confidence)
        return jsonify(data)

    return 'Picture info did not get saved.'


@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('test.jpg', 'rb')
    data = fileob.read()
    return data


if __name__ == '__main__':
    load_model()
    # model._make_predict_function()
    app.run(debug=False, port=5000)
