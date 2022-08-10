from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image

import numpy as np

model_name='skin_model.h5'
model=load_model(model_name)

def model_predict(imgPath,model):
	# loaded the image
	img=image.load_img(imgPath,target_size=(224,224))
	
	# convert this image into an array
	a=image.img_to_array(img)

	# 0 to 255 should be scaled to 0 to 1 - scaling
	a=a/255
	a=np.expand_dims(a,axis=0)

	# pass this array to model
	result=model.predict(a)
	result=np.argmax(result,axis=1)
	print (result)
	if result==0:
		result='Benign'
	elif result==1:
		result='Malignant'
	return (result)
	
	

# a name for our web app
app=Flask(__name__)

@app.route('/')
def index():
 return (render_template('index.html'))

@app.route('/predict',methods=['GET','POST'])
def upload():
	if request.method=='POST':
		f=request.files['file']

		basepath=os.path.dirname(__file__)
		file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
		f.save(file_path)

		preds=model_predict(file_path,model)
		result=preds
		return result
	return None


# run the web server
if __name__=='__main__':
	app.run(debug=True)