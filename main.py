from keras.models import model_from_json
import numpy as np

with open('catdog.json', 'r') as json_file:
    json_savedModel= json_file.read()
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.load_weights('catdog_weights.h5')

def pred(img_path):
	img = cv2.resize(img, (256, 256))
	cv2_imshow(img)
	img = np.reshape(img, (1, 256, 256, 3))
	if model_j.predict(img)[0][0]>model_j.predict(img)[0][1]:
	  print('It is a cat')
	print('cat, ', np.round(model_j.predict(img)[0][0]*100),'%')
	print('dog, ', np.round(model_j.predict(img)[0][1]*100), '%')

img_path = input()
pred(img_path)
