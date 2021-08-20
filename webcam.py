import numpy as np
from PIL import Image
import os

import transforms as transforms
from skimage.transform import resize
from models import *
import cv2
import yoloface
import conn_raspberry


cut_size = 44

transform_test = transforms.Compose([
	transforms.TenCrop(cut_size),
	transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def emotion_analyse(face_array, is_gray=False):
	if not is_gray:
		gray = rgb2gray(face_array)
		gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
	else:
		gray= np.array(face_array, dtype=np.uint8)
	img = gray[:, :, np.newaxis]
	img = np.concatenate((img, img, img), axis=2)
	img = Image.fromarray(img)
	inputs = transform_test(img)
	net.eval()
	ncrops, c, h, w = np.shape(inputs)
	inputs = inputs.view(-1, c, h, w)
	if torch.cuda.is_available():
		inputs = inputs.cuda(DEVICE)
	with torch.no_grad():
		inputs = Variable(inputs)
	outputs = net(inputs)
	outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
	score = F.softmax(outputs_avg, dim=0)
	return score.data.cpu().numpy()


def get_cropped_face(frame, faces):
	faces_frame = []
	for face in faces:
		faces_frame.append(frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]])
	return faces_frame


def get_cropeed_face_x1y1x2y2(frame, faces):
	faces_frame = []
	for face in faces:
		face = list(map(int, face))
		faces_frame.append(frame[face[1]:face[3], face[0]:face[2], ...])# __index__
	return faces_frame


def get_cropped_face_dlib(frame, faces):
	faces_frame = []
	for face in faces:
		faces_frame.append(frame[face.top():face.bottom(), face.left():face.right()])
	return faces_frame


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom, emotionsProba):
	# Draw a bounding box.
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
	text = 'face{:.2f}'.format(conf)
	# Display the label at the top of the bounding box
	label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, label_size[1])
	cv2.putText(frame, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
				(255, 255, 255), 1)
	maxIndex, secondIndex = arg_max_and_arg_second_submax(emotionsProba)
	cv2.putText(frame, "{}: {:.3f}".format(class_names[maxIndex], emotionsProba[maxIndex]), (left, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
				(255, 255, 255), 1)
	cv2.putText(frame, "{}: {:.3f}".format(class_names[secondIndex], emotionsProba[secondIndex]), (left, top - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.4,
				(255, 255, 255), 1)


def arg_max_and_arg_second_submax(nparray):
	temp = nparray.copy()
	index = np.argmax(temp)
	temp[index] = 0
	return index, np.argmax(temp)


def detect_faces_dlib(frame):
	import dlib
	detector = dlib.get_frontal_face_detector()
	faces = detector(frame, 0)
	return faces


def get_largest_face_location(boxes):
	#  0 1 2 3
	# x1y1x2y2
	max_area = (None, 0)  # (box, area)
	for box in boxes:
		area = abs(box[2] - box[0]) * abs(box[3] - box[1])
		if area > max_area[1]:
			max_area = (box, area)
	return max_area[0]


def load_fer2013_csv(data_file=r"./FER2013_VGG19/fer2013.csv"):
	""" loads fer2013.csv dataset
	# Arguments: data_file fer2013.csv
	# Returns: faces and emotions
			faces: shape (35887,48,48,1)
			emotions: are one-hot-encoded
	"""
	import pandas as pd
	data = pd.read_csv(data_file)
	pixels = data['pixels']
	emotions = data['emotion']
	usages = data['Usage']
	faces, fer_emotions = [], []
	for i, pixel in enumerate(pixels):
		if "Test" in usages[i]:
			img0 = np.array(list(map(int, pixel.split(" "))))
			np_img0 = np.asarray(img0)
			img0 = np_img0.reshape(48, 48)
			faces.append(img0)
			fer_emotions.append(emotions[i])
	return faces, fer_emotions


def get_max_face_x1y1wh(faces):
	max_face_info = (-1, None)
	for face in faces:
		area = face[2] * face[3]
		if area > max_face_info[0]:
			max_face_info = (area, face)
	return max_face_info[1]



#
# if __name__ == '__main__':
# 	if torch.cuda.is_available():
# 		gpuid = 0
# 		DEVICE = torch.device("cuda:%d" % gpuid)
# 		print("CUDA found! %s" % DEVICE)
# 	else:
# 		DEVICE = torch.device("cpu")
# 		gpuid = -1
# 		print("CUDA is not available, fall back to CPU.")
#
# 	net = VGG('VGG19')
# 	checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
# 	net.load_state_dict(checkpoint['net'])
# 	net.cuda(DEVICE)
# 	import glob
# 	import random
#
# 	filesL = []
# 	class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#
# 	import cv2
# 	print("loading fer2013 dataset...")
# 	faces, emotions = load_fer2013_csv()
# 	print("Loaded fer2013 dataset!")
# 	Y = emotions
#
# 	Y_predict = []
# 	for face in faces:
# 		emotionsProba = emotion_analyse(face, is_gray=True)
# 		y = np.argmax(emotionsProba)
# 		Y_predict.append(y)
# 	from sklearn.metrics import classification_report
# 	from sklearn.metrics import confusion_matrix
# 	import matplotlib.pyplot as plt
#
# 	#print(Y)
# 	#print(Y_predict)
# 	print(classification_report(Y, Y_predict))
# 	cf_matrix_svm = confusion_matrix(Y, Y_predict)
# 	fig, ax = plt.subplots()
# 	ax.matshow(cf_matrix_svm, cmap=plt.cm.Greens)
# 	im = ax.imshow(cf_matrix_svm, interpolation='nearest', cmap=plt.cm.Blues)
# 	ax.figure.colorbar(im, ax=ax)
# 	for x in range(len(cf_matrix_svm)):
# 		for y in range(len(cf_matrix_svm)):
# 			plt.annotate(cf_matrix_svm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
# 	plt.show()


if __name__ == '__main__':
	VIDEO_SOURCE = r"D:\Desktop\stephen 00_01_28-00_13_31.mkv"
	#VIDEO_SOURCE = 0

	#DETECTION_ALGORTHIM = 'DLIB'
	DETECTION_ALGORTHIM = 'YOLO'
	#DETECTION_ALGORTHIM = 'MTCNN'


	class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	cap = cv2.VideoCapture(VIDEO_SOURCE)
	#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
	#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

	net = VGG('VGG19')
	if torch.cuda.is_available():
		gpuid = 0
		DEVICE = torch.device("cuda:%d" % gpuid)
		print("CUDA found! %s" % DEVICE)
		checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
		net.load_state_dict(checkpoint['net'])
		net.cuda(DEVICE)
	else:
		DEVICE = torch.device("cpu")
		gpuid = -1
		print("CUDA is not available, fall back to CPU.")
		checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'), map_location=torch.device('cpu'))
		net.load_state_dict(checkpoint['net'])

	print("Connecting to RaspberryPi Through Internet...")
	conn_raspberry.setup_tcp_conn()

	if DETECTION_ALGORTHIM == "YOLO":
		while cap.isOpened():
			has_frame, frame = cap.read()
			faces, isFaceConfidences = yoloface.detect_faces(frame, showFrame=False)
			if len(faces) == 0:
				continue
			face = get_max_face_x1y1wh(faces)
			faces = [face]
			faces_frame = get_cropped_face(frame, faces)
			for i, face in enumerate(faces):
				emotionsProba = emotion_analyse(faces_frame[i])
				maxIndex, secondIndex = arg_max_and_arg_second_submax(emotionsProba)
				conn_raspberry.send_emotion(maxIndex)
				draw_predict(frame, 0.99, face[0], face[1], face[0] + face[2], face[1] + face[3], emotionsProba)
			cv2.imshow("camera", frame)
			cv2.waitKey(1)
		cap.release()
	elif DETECTION_ALGORTHIM == 'DLIB':
		while cap.isOpened():
			has_frame, frame = cap.read()
			faces_ltrb = detect_faces_dlib(frame)
			if len(faces_ltrb) != 0:
				faces_frame = get_cropped_face_dlib(frame, faces_ltrb)
				for i, d in enumerate(faces_ltrb):
					emotionsProba = emotion_analyse(faces_frame[i])
					draw_predict(frame, 1, d.left(), d.top(), d.right(), d.bottom(), emotionsProba)
			cv2.imshow("camera", frame)
			cv2.waitKey(1)
		cap.release()
	elif DETECTION_ALGORTHIM == 'MTCNN':
		from facenet_pytorch import MTCNN
		import matplotlib.pyplot as plt
		mtcnn = MTCNN(keep_all=True, device="cuda:0")
		while cap.isOpened():  # 720 1280 3
			has_frame, frame = cap.read()
			boxes, is_face_proba = mtcnn.detect(frame)
			if boxes is None:
				continue
			boxes = boxes.tolist()
			is_face_proba = is_face_proba.tolist()
			for i, face_proba in enumerate(is_face_proba):
				if face_proba < 0.95:
					del is_face_proba[i]
					del boxes[i]
			box = get_largest_face_location(boxes)
			print(box)
			if box is None:
				continue
			box = list(map(int, box))
			faces = get_cropeed_face_x1y1x2y2(frame, [box])
			face = faces[0]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			emotionsProba = emotion_analyse(face)
			maxIndex, secondIndex = arg_max_and_arg_second_submax(emotionsProba)
			conn_raspberry.send_emotion(maxIndex)
			
			draw_predict(frame, 0.99, box[0], box[1], box[2], box[3], emotionsProba)
			cv2.imshow("cam", frame)
			cv2.waitKey(1)
		cap.release()