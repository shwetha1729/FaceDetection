import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None
		self.chosen_wcs_predictions = None
		self.wc_errors=[]
		self.chosen_wc_ids=[]
		self.debug=False
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]

	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d imags.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			#debug
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations

#select weak classifiers to form a strong classifier
#after training, by calling self.sc_function(), a prediction can be made
#self.chosen_wcs should be assigned a value after self.train() finishes
#call Weak_Classifier.calc_error() in this function
#cache training results to self.visualizer for visualization
#
#
#detailed implementation is up to you
#consider caching partial results and using parallel computing
	def train_real(self,load_dir = None, save_dir = None):
		######################
		######## TODO ########
		######################
		n = self.data.shape[0]
		self.chosen_wcs_predictions = np.zeros((self.num_chosen_wc, n))
		if load_dir is not None and os.path.exists(load_dir):
			self.chosen_wcs = pickle.load(open(load_dir, 'rb'))

			n=self.data.shape[0]
			weights =np.array([1/n for i in range(n)])

			ada_picked_classifiers = []
			for alpha, wc in self.chosen_wcs:
				ada_picked_classifiers.append(self.weak_classifiers[wc.id])
			i=0
			self.debug = True if self.num_chosen_wc < 1000 else False
			for real_wc in tqdm(ada_picked_classifiers):
				error, prediction, thresholds, bin_pqs = real_wc.calc_error(weights,self.labels)
				real_wc.thresholds = thresholds
				real_wc.bin_pqs = bin_pqs
				#rem_classifiers[min_index].polarity = pol[min_index]
				predict_min = prediction
				self.chosen_wcs_predictions[i] = predict_min

				print('Minimum error:',error,'Minimum index: ',real_wc.id)
				alpha_i=1
				#wc_error_min, predict_min = rem_classifiers[min_index].calc_error(weights, self.labels)
				weights = weights * np.exp(-alpha_i * self.labels * predict_min)
				weights = weights / sum(weights)
				i+=1


	def train(self, load_dir = None, save_dir = None):
		######################
		######## TODO ########
		######################
		if self.style=='Real':
			self.train_real(load_dir,save_dir)
		else:
			self.debug = True if self.num_chosen_wc < 1000 else False
			if load_dir!=None and os.path.exists(load_dir):
				self.chosen_wcs = pickle.load(open( load_dir, 'rb'))
				self.wc_errors = pickle.load(open("errors_" + load_dir, 'rb'))
				self.chosen_wcs_predictions = pickle.load(open("preds_" +load_dir, 'rb'))



			else:
				n=self.data.shape[0]
				weights =np.array([1/n for i in range(n)])
				self.chosen_wcs=[]
				self.chosen_wcs_predictions = np.zeros((self.num_chosen_wc,n))
				k=1000 #if self.debug==False else 50
				t=5
				self.wc_errors={}
				min_index=-1
				for weak_classifier in self.weak_classifiers:
					weak_classifier.sorted_indices = np.argsort(weak_classifier.activations)

				rem_classifiers = [classifier for classifier in self.weak_classifiers]
				range_keys=[0,9,49,99] #if self.debug == False else [0,4,9,14,19]
				for i in tqdm(range(self.num_chosen_wc)):
					my_result_array = Parallel(n_jobs=self.num_cores)(
						delayed(wc.calc_error)(weights,self.labels) for wc in rem_classifiers
					)
					my_result_array = list(zip(*my_result_array))
					error = np.asarray(my_result_array[0])
					predict = np.asarray(my_result_array[1])
					threshold = np.asarray(my_result_array[2])
					pol = np.asarray(my_result_array[3])

					min_wc_error = min(error)
					min_index = np.argmin(error)
					rem_classifiers[min_index].threshold = threshold[min_index]
					rem_classifiers[min_index].polarity = pol[min_index]
					predict_min = predict[min_index]
					if i in range_keys:
						top_k_errors = np.partition(error, k)[:k]

						top_k_errors.sort()
						self.wc_errors[i] = top_k_errors

					predict_min = pol[min_index] * np.sign(rem_classifiers[min_index].activations - threshold[min_index])
					print('Minimum error:',min_wc_error,'Minimum index: ', min_index, 'in iteration ',i, 'classifier id', rem_classifiers[min_index])
					x = (1 - min_wc_error) / min_wc_error
					alpha_i = 0.5 * np.log(x)
					if self.style=='Real':
						alpha_i=1
					#wc_error_min, predict_min = rem_classifiers[min_index].calc_error(weights, self.labels)
					weights = weights * np.exp(-alpha_i * self.labels * predict_min)
					weights = weights / sum(weights)
					#print("Weightsum",sum(weights))
					p = rem_classifiers[min_index]
					self.chosen_wc_ids.append(p.id)
					print("id",id)
					#self.wc_errors[i] = min_wc_error
					self.chosen_wcs.append((alpha_i, p))
					self.chosen_wcs_predictions[i]=alpha_i*predict_min
					# print(i,":",self.chosen_wcs_predictions[i])
					del rem_classifiers[min_index]
				if save_dir is not None:
					pickle.dump(self.wc_errors,open("errors_"+save_dir, 'wb'))
					pickle.dump(self.chosen_wcs_predictions,open("preds_"+save_dir, 'wb'))
					pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))



	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))

	def face_detection(self, img, scale_step = 20):

		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################


		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)

		print('after nms:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #gree rectangular with line width 3

		return img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)

		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		predicts = np.array(predicts)
		wrong_patches = patches[np.where(predicts > 0), ...]

		return wrong_patches[0]

	def visualize(self,load_dir=None,save_dir=None):

		self.visualizer.labels = self.labels

		#self.visualizer.weak_classifier_accuracies = self.wc_errors
		cumsum= np.zeros((self.data.shape))
		cumsum = np.cumsum(self.chosen_wcs_predictions,axis=0)
		range_keys = [0, 9, 49, 99] #if self.debug==False else [0,4,9,14,19]
		for i in range_keys:
			self.visualizer.strong_classifier_scores[i+1] = cumsum[i]
		''''
		for j in range(20):
			#print("id:", self.chosen_wcs[j][1].id)
			self.visualizer.chosen_filters.append(self.filters[self.chosen_wcs[j][1].id])
		'''

		#print(len(self.visualizer.chosen_filters))
		#self.visualizer.draw_filters()
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		#self.visualizer.draw_wc_accuracies()
		#self.visualizer.draw_sc_errors(cumsum)
