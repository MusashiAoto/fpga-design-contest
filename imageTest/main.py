import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
K.set_session(sess)

import argparse
import os
import cv2
import numpy as np
import glob
# import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

import config as cf
from data_loader import DataLoader
#from fcn_seg import model
from InceptionResNetV2 import model
#from vgg16 import model
#from vgg16 import resNet50 as model
#from vgg16 import model_handmade as model
#from network import model

class Main_train():

	def __init__(self):
		pass

	def train(self):
		
		

		## Load network model
		self.net = model()
		#self.net = network.Mymodel()

		for layer in self.net.layers:
			layer.trainable = True
			print(layer.get_config())

		self.net.summary()
		#keras.utils.plot_model(self.net, to_file='model.png')

		optimizer = keras.optimizers.SGD(
			lr=cf.Learning_rate, decay=1e-5, momentum=0.9, nesterov=True)

		self.net.compile(loss='categorical_crossentropy',
					  optimizer=optimizer,
					  metrics=['accuracy'])

		## Prepare Training data
		dl_train = DataLoader(phase='Train', shuffle=True)

		## Prepare Test data
		dl_test = DataLoader(phase='Test', shuffle=True)
		test_imgs, test_gts = dl_test.get_minibatch(shuffle=False)

		## Start Train
		print('\n--------\nTraining Start!!')

		fname = os.path.join(cf.Save_dir,str(cf.time)+'_e'+str(cf.Step)+'_loss.txt')
		f = open(fname, 'w')
		f.write("Step,Train_L,Train_A,Test_L,Test_A,{}".format(os.linesep))

		for step in range(cf.Step):
			step += 1
			x, y = dl_train.get_minibatch(shuffle=True)
			#y = y.reshape((cf.Minibatch, -1, cf.Class_num))
			#history = self.net.fit(x=x, y=y, batch_size=cf.Minibatch, epochs=1, verbose=0)
			history = self.net.train_on_batch(x=x, y=y)
			train_loss = history[0]
			train_acc = history[1]

			if step % cf.SvStep == 0 or step == 1:
				x_test, y_test = dl_test.get_minibatch(shuffle=True)
				#y_test = y_test.reshape((y_test.shape[0], -1, cf.Class_num))
				loss, acc = self.net.evaluate(
					x=x_test, y=y_test, batch_size=len(x_test), verbose=0)
				print("Step: {:6d}, Train_L: {:.12f}, Train_A: {:.12f}, Loss: {:.12f}, Accuracy: {:.12f}".format(
					step, train_loss, train_acc, loss, acc))
				f.write("{},{},{},{},{}{}".format(step, train_loss, train_acc, loss, acc, os.linesep))
				f.flush()

			if step % cf.SvStep == 0 and cf.Step != step :
				ModelName = cf.time+'_e'+str(step)+'_CNN.h5'
				savePath = os.path.join(cf.Save_dir,cf.Ir_dir, ModelName)
				self.net.save(savePath)

		f.close()
		## Save trained model
		os.makedirs(cf.Save_dir, exist_ok=True)
		self.net.save(cf.Save_path)
		print('Model saved -> {}'.format(cf.Save_dir))


class Main_test():
	def __init__(self):
		pass


	def test(self):

		## Load network model
		self.net = model()

		self.net.load_weights(cf.Save_path)

		print('Test start !')

		table_gt_pred = np.zeros((cf.Class_num, cf.Class_num), dtype=np.int)

		img_paths = self.get_imagelist()

		pbar = tqdm(total=len(img_paths))

		for img_path in img_paths:
			img = self.get_image(img_path)
			gt = self.get_gt(img_path)

			scores = self.net.predict(x=img, batch_size=1, verbose=0)[0]

			pred_score = scores.max()
			pred_label = scores.argmax()

			table_gt_pred[gt, pred_label] += 1
			print('{} {}'.format(img_path, np.round(scores,4)*100))
			#sys.stdout.write("\r{}\r".format(img_path, gt, pred_label))
			pbar.update(1)

		print()
		for cls_ind in range(cf.Class_num):
			print(cf.Class_label[cls_ind], np.round(table_gt_pred[cls_ind], 3))



	def get_imagelist(self):
		dirs = cf.Test_dirs

		imgs = []

		for dir_path in dirs:
			img_list = glob.glob(dir_path + '/*')
			img_list.sort()
			imgs.extend(img_list)

		return imgs


	def get_image(self, img_path):

		img = cv2.imread(img_path).astype(np.float32)
		#img = cv2.resize(img, (cf.Width, cf.Height))

		if cf.Variable_input:
			out = np.zeros((cf.Max_side, cf.Max_side, 3), dtype=np.float32)
			longer_side = np.max(img.shape[:2])
			scaled_ratio = 1. * cf.Max_side / longer_side
			scaled_height = np.min([img.shape[0] * scaled_ratio, cf.Max_side]).astype(np.int)
			scaled_width = np.min([img.shape[1] * scaled_ratio, cf.Max_side]).astype(np.int)
			img = cv2.resize(img, (scaled_width, scaled_height))
			out[:scaled_height, :scaled_width, :] = img
			img = out
		else:
			scaled_height = cf.Height
			scaled_width = cf.Width
			img = cv2.resize(img, (scaled_width, scaled_height))

		img = img[:, :, (2,1,0)]
		img = img[np.newaxis, :]
		img = img / 255.

		return img

	def get_gt(self, img_path):

		for ind, cls in enumerate(cf.Class_label):
			if cls in img_path:
				return ind
		return cf.Class_num - 1
		#raise Exception("Class label Error {}".format(img_path))


def arg_parse():
	parser = argparse.ArgumentParser(description='CNN implemented with Keras')
	parser.add_argument('--train', dest='train', action='store_true')
	parser.add_argument('--test', dest='test', action='store_true')
	parser.add_argument('-ti','--trainImagePath')
	parser.add_argument('-si','--testImagePath')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = arg_parse()
	cf.setImage(args.trainImagePath,args.testImagePath)

	if args.train:
		main = Main_train()
		main.train()
	if args.test:
		main = Main_test()
		main.test()
