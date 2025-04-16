from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('weights.h5')

# Upload history to display
upload_history = []

def preprocess_image(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html', img_path=None, result=None, history=upload_history)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return render_template('index.html', error="No image provided", history=upload_history)

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', error="No selected file", history=upload_history)

    # Save the uploaded image
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)

    # Preprocess and predict
    img = preprocess_image(Image.open(filepath))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:
        result = {
            "message": "⚠️ This uploaded image looks like melanoma cancer.",
            "is_cancer": True,
            "confidence": confidence,
            "symptoms": [
                "Unusual moles or spots",
                "Changes in mole size, shape or color",
                "Itching or bleeding from mole",
                "Irregular borders"
            ],
            "advice": "Please consult a dermatologist immediately.",
            "tips": [
                "Use sunscreen daily with SPF 30+",
                "Avoid tanning beds",
                "Regularly check your skin",
                "Wear protective clothing"
            ]
        }
    else:
        result = {
            "message": "✅ This uploaded image does not show signs of skin cancer.",
            "is_cancer": False,
            "confidence": 100 - confidence,
            "symptoms": [],
            "advice": "Keep monitoring your skin. Seek medical help if you notice changes.",
            "tips": [
                "Stay hydrated and maintain healthy skin",
                "Avoid prolonged sun exposure",
                "Use moisturizers and gentle cleansers"
            ]
        }

    # Save to history
    upload_history.insert(0, {
        "img_path": '/' + filepath,
        "result": result["message"],
        "confidence": result["confidence"]
    })
    if len(upload_history) > 5:
        upload_history.pop()

    return render_template('index.html', img_path='/' + filepath, result=result, history=upload_history)

if __name__ == '__main__':
    app.run(debug=True)
