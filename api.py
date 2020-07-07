import PIL
from PIL import Image
PIL.PILLOW_VERSION = PIL.__version__
from fastai.vision import *
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, Response
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Console(Resource):
    def get(self):
        content = str(open('index.html').read())
        return Response(content, mimetype="text/html")
    
    def put(self):
        learn = load_learner('model')
        
        fd = urlopen(request.args.get('url'))
        image_file = io.BytesIO(fd.read())
        img = open_image(image_file)
        
        ypred = learn.predict(img)
        return str(ypred[0])
    
    def post(self):
        img = open_image(request.files['image'])
        
        learn = load_learner('model')
        ypred = learn.predict(img)
        return "Hummm... 🤔 Je vois une " + str(ypred[0])

api.add_resource(Console, '/console')

if __name__ == '__main__':
     app.run(port='5002')
