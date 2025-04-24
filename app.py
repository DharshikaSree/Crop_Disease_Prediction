from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
# from flask_caching import Cache

    # Prediction logic


app = Flask(__name__) 

# Load the model
model = tf.keras.models.load_model('model.h5')

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

#class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Late_blight', 'Tomato___healthy']
class_names = [
 'Maize___Blight',
 'Maize___Common_Rust',
 'Maize___Gray_Leaf_Spot',
 'Maize___Healthy',
 'Rice___Bacterial leaf blight',
 'Rice___Brown spot',
 'Rice___Leaf smut',
 'Soybean___Caterpillar',
 'Soybean___Diabrotica_speciosa',
 'Soybean___Healthy',
 'Wheat__Healthy',
 'Wheat__Septoria',
 'Wheat__stripe_rust']

BATCH_SIZE = 32
IMAGE_SIZE = 255
CHANNEL = 3
EPOCHS = 10

# Define a dictionary to map diseases to treatment suggestions
disease_treatments = {
    'Rice___Bacterial leaf blight': [
        "Use resistant rice varieties.",
        "Avoid excessive nitrogen fertilizer.",
        "Apply copper-based bactericides if needed."
    ],
    'Rice___Brown spot': [
        "Use certified disease-free seeds.",
        "Apply fungicides like azoxystrobin or mancozeb.",
        "Ensure proper field drainage and balanced fertilization."
    ],
    'Rice___Leaf smut': [
        "Use resistant varieties and seed treatment with fungicides.",
        "Apply tricyclazole or isoprothiolane-based fungicides at early infection stages.",
        "Maintain proper water management to reduce stress."
    ],
    'Wheat__Septoria': [
        "Use resistant wheat varieties whenever possible.",
        "Apply fungicides like triazoles (e.g., tebuconazole, prothioconazole) or strobilurins."
    ],
    'Wheat__Stripe_rust': [
        "Grow resistant wheat cultivars.",
        "Monitor fields early and apply fungicides like triazoles or strobilurins when needed.",
        "Ensure proper nitrogen management to avoid excessive lush growth."
    ],
    'Wheat__Healthy': [
        "No treatment needed.",
        "Regularly monitor fields and use best agricultural practices."
    ],

    'Maize___Gray_leaf_spot': [
        "Use resistant maize hybrids.",
        "Apply fungicides like pyraclostrobin or azoxystrobin if necessary.",
        "Ensure proper field sanitation and avoid overhead irrigation."
    ],
    'Maize___Common_rust': [
        "Plant resistant maize varieties.",
        "Fungicide sprays containing triazoles or strobilurins can help in severe cases.",
        "Maintain crop rotation and avoid planting maize continuously in the same field."
    ],
    'Maize___Blight': [
        "Use disease-resistant hybrids.",
        "Apply fungicides like propiconazole or mancozeb at early infection stages.",
        "Practice crop rotation and remove crop debris after harvest."
    ],
    'Maize___Healthy': [
        "No treatment needed.",
        "Monitor crops regularly and maintain optimal growing conditions."
    ],
     'Soybean___Caterpillar': [
        "Use biological control methods like Bacillus thuringiensis (Bt).",
        "Apply insecticides such as spinosad or pyrethroids if infestations are severe.",
        "Encourage natural predators like birds and parasitic wasps."
    ],
    
    'Soybean___Diabrotica_speciosa': [
        "Rotate crops to prevent larvae from developing in the soil.",
        "Use seed treatments with systemic insecticides like thiamethoxam or imidacloprid.",
        "Apply foliar insecticides if adult beetle populations exceed economic thresholds."
    ],
    
    'Soybean___Healthy': [
        "No treatment needed.",
        "Monitor fields regularly for early pest detection.",
        "Ensure proper soil health and balanced fertilization to maintain plant resistance."
    ],
    # 'Rice___Healthy': [
    #     "No treatment needed.",
    #     "Maintain good crop management practices to prevent diseases."
    # ],
    # 'Potato___Early_blight': [
    #     "Remove and destroy infected leaves.", 
    #     "Apply fungicides like chlorothalonil or mancozeb."
    # ],
    # 'Potato___Late_blight': [
    #     "Remove and destroy infected plants.", 
    #     "Apply fungicides like metalaxyl or azoxystrobin."
    # ],
    # 'Potato___healthy': [
    #     "No treatment needed."
    # ],
    # 'Tomato___Late_blight' : [
    #     "Prevent late blight by choosing resistant varieties and providing good air circulation.", 
    #     "Act fast if you see any signs of the disease, removing infected parts immediately."
    # ],
    # 'Tomato___healthy': [
    #     "No treatment needed."
    # ]
    # Add more diseases and treatments here
}


# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('home.html')


# Route to the home page
@app.route('/second', methods=['GET', 'POST'])
def second():
    if request.method == 'POST':
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template('index.html', message="No file part")

        file = request.files['file']  # Corrected line

        # If the user does not select a file, browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        # If the file is allowed and has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Read the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            # Predict using the loaded model
            predicted_class, confidence = predict(img)


            # Get the treatment suggestion for the predicted disease
            treatment_suggestion = disease_treatments.get(predicted_class, ["No treatment suggestion available."])

        # Render the template with the uploaded image, actual and predicted labels, and confidence. 
        # return render_template('index.html', image_path=filepath, actual_label=predicted_class, predicted_label=predicted_class, confidence=confidence, treatments=treatment_suggestion)


        # Render the template with the uploaded image, actual and predicted labels, confidence, and treatment suggestion
        return render_template('index.html', image_path=filepath, predicted_label=predicted_class, confidence=confidence, treatments=treatment_suggestion)

   # return render_template('index.html')
    return render_template('index.html', message='Upload an image')

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


if __name__ == "__main__":
    app.run(debug=True)