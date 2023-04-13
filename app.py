from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
import pyttsx3

app = Flask(__name__)

IMG_SIZE = 200

model = tf.keras.models.load_model('Final_CNN model.h5')
classes = ['Ten Rupees', 'Twenty Rupees', 'Fifty Rupees', 'Hundred Rupees', 'Two Hundred Rupees', 'Five Hundred Rupees', 'Two Thousand Rupees']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('home.html', error='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('home.html', error='No file selected')
        if file:
            # Preprocess the uploaded image or audio file
            if file.content_type.startswith('image/'):
                img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                # Get the predicted denomination
                pred = model.predict(img)[0]
                class_idx = np.argmax(pred)
                denomination = classes[class_idx]
                # Generate the audio output
                engine = pyttsx3.init()
                engine.say('The denomination is ' + denomination)
                engine.runAndWait()
                # Make a prediction on a single image
                prediction = model.predict(img)
                class_idx = np.argmax(prediction, axis=1)[0]
                class_name = classes[class_idx]
                # Speak the prediction aloud
                engine.say("Prediction: " + class_name)
                engine.runAndWait()
            elif file.content_type.startswith('audio/'):
                # Use a speech recognition API to convert the audio into text
                # Replace the following line with your own code
                text = 'fifty'
                if text in classes:
                    denomination = text
                else:
                    return render_template('home.html', error='Unable to recognize denomination')
            else:
                return render_template('home.html', error='Invalid file type')
            return render_template('home.html', success=True)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
