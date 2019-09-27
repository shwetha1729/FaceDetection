import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *

def main():
	#flag for debugging
	flag_subset = True
	hard_neg = False
	boosting_type = 'Ada' #'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

	#data configurations
	pos_data_dir = 'newface16'
	neg_data_dir = 'nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	data = integrate_images(normalize(data))

	#number of bins for boosting
	num_bins = 25

	#number of cpus for parallel computing
	num_cores = 8 if not flag_subset else 1 #always use 1 when debugging
	
	#create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

	#create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])
	
	#create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	#calculate filter values for all training images
	start = time.clock()
	#boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	boost.train(chosen_wc_cache_dir, chosen_wc_cache_dir)

	boost.visualize()
	#face detection
	original_img = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
	result_img = boost.face_detection(original_img)
	cv2.imwrite('Result_img_%s.png' % boosting_type, result_img)

	if hard_neg == True:
		#neg_img

		start = time.clock()
		for i in range(1,3):
			import os
			sd = os.path.join(os.path.curdir,'Testing_images')
			print(os.listdir(sd))
			ipath =  os.path.join(sd, 'Non_Face_'+str(i)+'.jpg')
			print(ipath)
			img = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
			print("i",i)
			new_data = boost.get_hard_negative_patches(img)
			#boost.calculate_training_activations(, )
			load_dir = "hard_neg_"+str(i)+act_cache_dir
			save_dir = "hard_neg_"+str(i)+act_cache_dir
			print('Calculate activations for %d weak classifiers, using %d imags.' % (
			len(boost.weak_classifiers), new_data.shape[0]))
			if load_dir is not None and os.path.exists(load_dir):
				print('[Find cached activations, %s loading...]' % load_dir)
				wc_activations = np.load(load_dir)
			else:
				if boost.num_cores == 1:
					wc_activations = [wc.apply_filter(new_data) for wc in boost.weak_classifiers]
				else:
					from joblib import Parallel, delayed
					wc_activations = Parallel(n_jobs=boost.num_cores)(
						delayed(wc.apply_filter)(new_data) for wc in boost.weak_classifiers)
				wc_activations = np.array(wc_activations)
				if save_dir is not None:
					print('Writing results to disk...')
					np.save(save_dir, wc_activations)
					print('[Saved calculated activations to %s]' % save_dir)
			for wc in boost.weak_classifiers:
				wc.activations = np.concatenate((wc.activations, wc_activations[wc.id, :]))
			boost.data = np.concatenate((boost.data, new_data))
			newlabels = np.ones((new_data.shape[0]))
			newlabels = newlabels * -1
		boost.labels = np.concatenate((boost.labels,newlabels))
		end = time.clock()


		print('%f seconds for activation calculation' % (end - start))

		boost.train(chosen_wc_cache_dir, chosen_wc_cache_dir)

		boost.visualize()

def main_real():
	flag_subset = False
	boosting_type = 'Real'  # 'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

	# data configurations
	pos_data_dir = 'newface16'
	neg_data_dir = 'nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	data = integrate_images(normalize(data))

	# number of bins for boosting
	num_bins = 25

	# number of cpus for parallel computing
	num_cores = 8 if not flag_subset else 1  # always use 1 when debugging

	# create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

	# create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])

	# create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	# calculate filter values for all training images
	start = time.clock()
	# boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	boost.train(chosen_wc_cache_dir, None)

	boost.visualize()
	original_img = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
	result_img = boost.face_detection(original_img)
	cv2.imwrite('Result_img_%s.png' % boosting_type, result_img)
	original_img = cv2.imread('./Testing_Images/Face_2.jpg', cv2.IMREAD_GRAYSCALE)
	result_img = boost.face_detection(original_img)
	cv2.imwrite('Result_img_%s.png' % boosting_type, result_img)
	original_img = cv2.imread('./Testing_Images/Face_3.jpg', cv2.IMREAD_GRAYSCALE)
	result_img = boost.face_detection(original_img)
	cv2.imwrite('Result_img_%s.png' % boosting_type, result_img)

if __name__ == '__main__':
	#main()
	main_real()
