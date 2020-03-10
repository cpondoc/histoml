import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import *
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from models import *

# Important Global Variables
UPLOAD_FOLDER = 'uploads/'
ASSET_FOLDER = 'assets/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
root_dir = "http://127.0.0.1:5000/"
model_nombre = "VGG-19"

# Setting up the App
app = Flask(__name__, static_url_path = '/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ASSET_FOLDER'] = ASSET_FOLDER

# Defining Pre-Trained Models
def unfreeze_layers(model_name, conv_base):
    
    # Unfreezing all
    conv_base.trainable = True
    
    # Case for VGG-16
    if (model_name == "VGG-16" or model_name == "VGG-19"):
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
    
    # Case for Xception            
    elif (model_name == "Xception"):
        for layer in conv_base.layers[:10]:
            layer.trainable = False
    
    # Case for ResNetV2        
    elif (model_name == "ResNetV2"):
        for layer in conv_base.layers[:26]:
            layer.trainable = False
    
    # Case for InceptionV3
    elif (model_name == "InceptionV3"):
        for layer in conv_base.layers[:249]:
            layer.trainable = False
            
    # Case for InceptionResNetV2
    elif (model_name == "InceptionResNetV2"):
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block8_9_mixed':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
                
    # Case for MobileNet
    elif (model_name == "MobileNet" or model_name == "MobileNetV2"):
        for layer in conv_base.layers[:10]:
            layer.trainable = False
    
    # Case for DenseNet
    elif (model_name == "DenseNet"):
        for layer in conv_base.layers[:15]:
            layer.trainable = False
    
    # Case for NASNetLarge
    elif (model_name == "NASNetLarge"):
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'activation_253':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

# Looking at an allowed files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(request.form['model_name'])
            model_selected = str(request.form['model_name'])
            return redirect(url_for('predict_file', filename=filename, model_name=model_selected))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/assets/<filename>')
def asset_file(filename):
    return send_from_directory(app.config['ASSET_FOLDER'],
                               filename)

@app.route('/predict/<model_name>/<filename>')
def predict_file(filename, model_name):
    file_name = root_dir + "uploads/" + filename
    conv_base = return_pretrained(model_name)
    model = build_model(conv_base)
    unfreeze_layers("", conv_base)
    model = load_model("weights/" + model_name + ".h5")
    print("This is working so far!")
    # Loading the image in
    test_image = image.load_img("uploads/" + filename, target_size=(425, 256))
    
    # Casting the image into an array
    test_image = image.img_to_array(test_image)
    
    # Expanding the dimensions of the image
    test_image = np.expand_dims(test_image, axis = 0)
    
    # Dividing by 255
    test_image = test_image / 255
    
    """ Uncomment once machine is freed up """
    
    # Predicting the type of the test image
    result = model.predict(test_image)

    # Checking what the network guesses
    if (result[0] <= .5):
        guess = "Carcinoma"
        if (result[0] < .25):
            confidence_level = "Strong"
        else:
            confidence_level = "Weak"
    else:
        guess = "Sarcoma"
        if (result[0] > .75):
            confidence_level = "Strong"
        else:
            confidence_level = "Weak"

    print(result)
    return render_template('prediction.html', model_name = model_name, file_name = file_name, prediction = guess, confidence = result, level = confidence_level)