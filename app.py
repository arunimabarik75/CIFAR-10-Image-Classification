import numpy as np
from flask import Flask, render_template, request, session
import os
import pickle
import cv2

# Loading Model
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def loadFromFile(filename):

    f = open(filename, 'rb')
    theSet = pickle.load(f, encoding='latin1')
    f.close()

    return theSet

# Convert from image to numpy array


def convertImage(origImage):

    image = np.reshape(origImage, (-1, 3, 32, 32))
    image = np.transpose(image, (0, 2, 3, 1))

    return image


# HOG parameters
winSize = 32
blockSize = 12
blockStride = 4
cellSize = 4
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = True
hog = cv2.HOGDescriptor((winSize, winSize), (blockSize, blockSize), (blockStride, blockStride), (cellSize, cellSize),
                        nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

# Calculate Hog features


def calcHOG(image):
    hogDescriptor = hog.compute(image)
    hogDescriptor = np.squeeze(hogDescriptor)
    return hogDescriptor

# Load the models from pickle files


pca = pickle.load(open('pca_model.pkl', 'rb'))
svm = pickle.load(open('svm_model.pkl', 'rb'))

# Find image class


def classifyImage(testImage):

    # testImage = convertImage(testImage)
    testHogDescriptor = calcHOG(testImage)
    testHogProjected = pca.transform(testHogDescriptor.reshape(1, -1))
    testResponse = svm.predict(testHogProjected)

    return testResponse

# Class values


label = {}
label[0] = 'airplane'
label[1] = 'automobile'
label[2] = 'bird'
label[3] = 'cat'
label[4] = 'deer'
label[5] = 'dog'
label[6] = 'frog'
label[7] = 'horse'
label[8] = 'ship'
label[9] = 'truck'

UPLOAD_FOLDER = os.path.join('static/tempFiles')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route('/')
def home():
    return render_template('index.html', message="", color='red')


@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        if request.form['btn'] == 'submit':
            uploaded_img = request.files['uploaded-file']
            uploaded_img.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'img.png'))
            path = os.path.join('static/tempFiles', 'img.png')
            pth = 'tempFiles/img.png'
            testImage = cv2.imread(path)
            testImage = cv2.resize(testImage, (32, 32))
            id = classifyImage(testImage)
            id = id[0]
            output = label.get(id)
            print(output)
            return render_template('prediction.html', class_image='Image might have: '+output, p=pth)
        elif request.form['btn'] == 'return':
            return render_template('index.html', color='red')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
