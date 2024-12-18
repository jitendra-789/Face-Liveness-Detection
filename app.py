from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import numpy as npdfg
import io
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
yolo_model = YOLO("anil.pt")

# Load Keras model for real or spoof classification
keras_model = load_model("liveness.keras")

# Compile the Keras model if necessary
keras_model.compile(
    optimizer='adam'
    loss='binary_crossentropy',
    metrics=['accuracy']
)

@app.route("/")
def root():
    try:
        with open("index.html") as file:
            return file.read()
    except FileNotFoundError:
        return "index.html not found", 404

@app.route("/success")
def success():
    try:
        with open("success.html") as file:
            return file.read()
    except FileNotFoundError:
        return "success.html not found", 404


@app.route("/detect", methods=["POST"])
def detect():
    if request.is_json:
        # Handle JSON payload from webcam capture
        data = request.get_json()
        image_data = data.get('image_data')
        if not image_data:
            return Response(
                "No image data provided",  
                mimetype='text/plain'
            ), 400
        # Decode base64 image data
        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    elif "image_file" in request.files:
        # Handle file upload
        buf = request.files["image_file"]
        image = Image.open(buf.stream)
    else:
        return Response(
            "No image data or file provided",  
            mimetype='text/plain'
        ), 400

    try:
        boxes, safety_status = detect_objects_on_image(image)
        
        # Check if exactly one person is detected
        if len([box for box in boxes if box[4] in ["person", "Person"]]) == 1:
            # Preprocess the image for Keras model
            preprocessed_image = preprocess_image(image)
            # Predict using Keras model
            real_or_spoof = keras_model.predict(np.expand_dims(preprocessed_image, axis=0))[0]
            # Determine and append real or spoof status
            real_or_spoof_status = 'Real' if real_or_spoof[0] > 0.5 else 'Spoof'  
            safety_status += f" | Status: {real_or_spoof_status}"

        return Response(
            f"{safety_status}",  
            mimetype='text/plain'
        )
        
    except Exception as e:
        return Response(
            f"Error: {str(e)}",  
            mimetype='text/plain'
        ), 500

def detect_objects_on_image(image):
    results = yolo_model.predict(image)
    result = results[0]
    output = []
    person_count = 0
    
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        class_name = result.names[class_id]
        output.append([
            x1, y1, x2, y2, class_name, prob
        ])

        if class_name in ["person", "Person"]:
            person_count += 1

    if person_count > 1:
        safety_status = f"Multiple persons detected: {person_count}"
    elif(person_count==0):
        safety_status=f"No persons detected."
    else:
        safety_status = f"Result: {person_count} person detected"

    return output, safety_status

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  
    return image_array

if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=5000)
