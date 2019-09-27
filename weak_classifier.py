from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed


class Weak_Classifier(ABC):
	# initialize a harr filter with the positive and negative rects
	# rects are in the form of [x1, y1, x2, y2] 0-index
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		self.id = id
		self.plus_rects = plus_rects
		self.minus_rects = minus_rects
		self.num_bins = num_bins

		self.activations = None

	# take in one integrated image and return the value after applying the image
	# integrated_image is a 2D np array
	# return value is the number BEFORE polarity is applied
	def apply_filter2image(self, integrated_image):
		pos = 0
		for rect in self.plus_rects:
			rect = [int(n) for n in rect]
			pos += integrated_image[rect[3], rect[2]] \
				   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1]) \
				   - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]]) \
				   - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		neg = 0
		for rect in self.minus_rects:
			rect = [int(n) for n in rect]
			neg += integrated_image[rect[3], rect[2]] \
				   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1]) \
				   - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]]) \
				   - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		return pos - neg

	# take in a list of integrated images and calculate values for each image
	# integrated images are passed in as a 3-D np-array
	# calculate activations for all images BEFORE polarity is applied
	# only need to be called once
	def apply_filter(self, integrated_images):
		values = []
		for idx in range(integrated_images.shape[0]):
			values.append(self.apply_filter2image(integrated_images[idx, ...]))
		if (self.id + 1) % 100 == 0:
			print('Weak Classifier No. %d has finished applying, activation length = %s' % (self.id + 1, len(values)))
		return values

	# using this function to compute the error of
	# applying this weak classifier to the dataset given current weights
	# return the error and potentially other identifier of this weak classifier
	# detailed implementation is up you and depends
	# your implementation of Boosting_Classifier.train()
	@abstractmethod
	def calc_error(self, weights, labels):
		pass

	@abstractmethod
	def predict_image(self, integrated_image):
		pass


class Ada_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.sorted_indices = None
		self.polarity = None
		self.threshold = None

	def calc_error(self, weights, labels):
		# https://courses.cs.washington.edu/courses/cse576/17sp/notes/FaceDetection17.pdf
		n = weights.shape
		#errors = np.zeros(weights.shape[0])
		#polarities = np.zeros(weights.shape)
		new_weights = weights[self.sorted_indices]
		new_labels = labels[self.sorted_indices]
		pos_mask = new_labels==1
		neg_mask = new_labels==-1

		#print("new wts",new_weights.shape)
		FS = np.cumsum(np.where(pos_mask, new_weights,0))
		BG = np.cumsum(np.where(neg_mask, new_weights,0))

		AFS = FS[-1]
		ABG = BG[-1]

		right = FS + ABG - BG
		left = BG + AFS - FS
		errors=np.minimum(left,right)
		# for index in self.sorted_indices:
		# 	if labels[index] == 1:
		# 		FS += weights[index]
		# 	else:
		# 		BG += weights[index]
		# 	a = BG + (AFS - FS)
		# 	b = FS + (ABG - BG)
		# 	# print("errors.shape ",errors.shape)
		# 	errors[index] = min(a, b)

		min_index 	   = np.argmin(errors)
		self.threshold = self.activations[self.sorted_indices[min_index]]
		self.polarity  = -1 if left[min_index] < right[min_index] else 1
		final_error    = errors[min_index]

		# print('Final error in calc_error:', final_error);
		# print('Activations in calc_error:', self.activations);
		# print('Weights in calc_error:', weights);
		# print('Labels in calc_error:', labels);
		# print('Sorted indices in calc_error:',self.sorted_indices );
		# print('Errors in calc_error:', errors);
		# exit()

		#predictions=np.zeros(n)

		predictions = self.polarity * np.sign(self.activations - self.threshold)
		return final_error,predictions,self.threshold,self.polarity


	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		return self.polarity * np.sign(value - self.threshold)


class Real_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.thresholds = None  # this is different from threshold in ada_weak_classifier, think about it
		self.bin_pqs = None
		self.train_assignment = None
		self.sorted_activations = None
	def calc_error(self, weights, labels):
		######################
		######## TODO ########
		######################
		n=weights.shape[0]
		# find max(activations) and min(activation) -> range of activations
		max_act = max(self.activations)
		min_act = min(self.activations)
		step_size = (max_act-min_act)/self.num_bins
		p_b = np.zeros(self.num_bins)
		q_b = np.zeros(self.num_bins)
		bin_left=[(min_act+i*step_size) for i in range(0,self.num_bins)]
		bin_right=[left + step_size for left in bin_left]
		for bin in range(self.num_bins):
			is_in_bin = [True if act_i>bin_left[bin] and act_i<=bin_right[bin] else False for act_i in self.activations]
			is_in_bin = np.array(is_in_bin)
			p_b[bin] = sum(weights[np.logical_and(is_in_bin,labels== 1)])
			q_b[bin] = sum(weights[np.logical_and(is_in_bin,labels==-1)])
		loss = 2*np.sum(np.sqrt(p_b*q_b))
		p_b[p_b == 0] = 1e-7
		q_b[q_b == 0] = 1e-7

		self.bin_pqs=np.zeros((2,self.num_bins))
		self.bin_pqs[0] = p_b
		self.bin_pqs[1] = q_b
		predict=[]
		self.thresholds = np.array(bin_right[:self.num_bins-1])
		for i in range(n):
			bin_idx = np.sum(self.thresholds < self.activations[i])
			#print("bin_idx",bin_idx)
			predict.append(0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx]))
		return loss,predict,self.thresholds,self.bin_pqs
		# for every bin b
		#   find p(b) and q(b)
		#       p(b) = sum of weights(positive data points in bin b)
		#       q(b) = sum of weights(negative data points in bin b)
		#  return loss = 2*np.sum(np.sqrt(pq))


	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		bin_idx = np.sum(self.thresholds < value)
		return 0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx])


def main():
	plus_rects = [(1, 2, 3, 4)]
	minus_rects = [(4, 5, 6, 7)]
	num_bins = 50
	ada_hf = Ada_Weak_Classifier(1, plus_rects, minus_rects, num_bins)
	real_hf = Real_Weak_Classifier(2, plus_rects, minus_rects, num_bins)


if __name__ == '__main__':
	main()
