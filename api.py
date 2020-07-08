import PIL
from PIL import Image
PIL.PILLOW_VERSION = PIL.__version__
from fastai.vision import *
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, Response
from flask_restful import Resource, Api

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# convertion musique en spectrograme
def audio2image(filename):
    x, sample_rate = librosa.load(filename, offset=30,duration=30)
    
    n_fft = 1024
    hop_length = 256
    n_mels = 40
    fmin = 20
    fmax = sample_rate / 2 
    
    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft, 
                                                    hop_length=hop_length, 
                                                    n_mels=n_mels, power=2.0, 
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    plt.imsave('spectrogram.png', mel_spec_db)

# API
app = Flask(__name__)
api = Api(app)

# Console
class Console(Resource):
    def get(self):
        content = str(open('console.html').read())
        return Response(content, mimetype="text/html")
    
    def post(self):
        img = open_image(request.files['image'])
        
        learn = load_learner('model_console')
        ypred = learn.predict(img)
        print()
        return "Hummm... Je vois une " + str(ypred[0]) + " (¬‿¬)"

api.add_resource(Console, '/console')

# is edm or classical
class IsClassicalOrEdm(Resource):
    def get(self):
        content = str(open('is_classical_or_edm.html').read())
        return Response(content, mimetype="text/html")
    
    def post(self):
        file = request.files.getlist('file')
        request.files['music'].save('music.mp3')
        
        learn = load_learner('model_is_edm')
        audio2image('music.mp3')
        img = open_image('spectrogram.png')
        ypred = learn.predict(img)
        
        os.remove('spectrogram.png')
        os.remove('music.mp3')

        print(ypred[0])
        return "It's " + str(ypred[0]) + " ! (¬‿¬)"

api.add_resource(IsClassicalOrEdm, '/isClassicalOrEdm')

# run
if __name__ == '__main__':
    app.run(host= '0.0.0.0', port='5002')
