'''
This code is a modified version of https://github.com/nmrksic/counter-fitting for creating emotional embeddings.
Developed by Armin Seyeditabari & Narges Tabari.
'''
import ConfigParser
import numpy
import sys
import time
import random 
import math
import os
from copy import deepcopy
from numpy.linalg import norm
from numpy import dot
# from scipy.stats import spearmanr

class ExperimentRun:
	"""
	EmperimentRun stores hyperparameters and datasets required for this analysis. 
	"""
	def __init__(self, config_filepath):
		"""
		The config file stores the location of the pre-trained word-verctors and vocabularies we use
		in this project, and the parameters. 

		To start, first supply the config file. This function reads that file. 
		"""
		self.config = ConfigParser.RawConfigParser()
		try:
			self.config.read(config_filepath)
		except:
			print "Could not read config file from", config_filepath
			return None

		self.pretrained_vectors_filepath = self.config.get("data", "pretrained_vectors_filepath")
		vocabulary_filepath = self.config.get("data", "vocabulary_filepath")
		
		vocabulary = []
		with open(vocabulary_filepath, "r+") as f_in:
			for line in f_in:
				vocabulary.append(line.strip())

		vocabulary = set(vocabulary)
		
		# load pre-trained word-vectors and initialize their (restricted) vocabulary. 
		self.pretrained_word_vectors = load_word_vectors(self.pretrained_vectors_filepath, vocabulary)

		# Make sure word-vectores are loaded:
		if not self.pretrained_word_vectors:
			return

		self.vocabulary = set(self.pretrained_word_vectors.keys())

		# filenames for synonyms and antonyms 
		synonym_list = self.config.get("data", "synonyms").replace("[","").replace("]", "").replace(" ", "").split(",")
		antonym_list = self.config.get("data", "antonyms").replace("[","").replace("]", "").replace(" ", "").split(",")

		self.synonyms = set()
		self.antonyms = set()

		# and we then have all the information to collect all the linguistic constraints:
		for syn_filepath in synonym_list:
			self.synonyms = self.synonyms | load_constraints(syn_filepath, self.vocabulary)

		for ant_filepath in antonym_list:
			self.antonyms = self.antonyms | load_constraints(ant_filepath, self.vocabulary)

		# load the experiment hyperparameters:
		self.load_experiment_hyperparameters()
	

	def load_experiment_hyperparameters(self):
		"""
		loading the hyperparameters. For information on what these parameters are, please refer to the paper.
		"""
		self.hyper_k1 = self.config.getfloat("hyperparameters", "hyper_k1")
		self.hyper_k2 = self.config.getfloat("hyperparameters", "hyper_k2") 
		self.hyper_k3 = self.config.getfloat("hyperparameters", "hyper_k3") 
		self.delta    = self.config.getfloat("hyperparameters", "delta")
		self.gamma    = self.config.getfloat("hyperparameters", "gamma")
		self.rho      = self.config.getfloat("hyperparameters", "rho")


		print "\nHyperparameters of this experiment (k_1, k_2, k_3, delta, gamma, rho):", \
			   self.hyper_k1, self.hyper_k2, self.hyper_k3, self.delta, self.gamma, self.rho


def load_word_vectors(file_destination, vocabulary):
	"""
	Loading the word vectors form the file path, then prints size, and vector dimentionality. 
	"""
	print "Loading pretrained word vectors from", file_destination
	word_dictionary = {}

	try:
		with open(file_destination, "r") as f:
			for line in f:
				line = line.split(" ", 1)	
				key = line[0].lower()
				if key in vocabulary:	
					word_dictionary[key] = numpy.fromstring(line[1], dtype="float32", sep=" ")
	except:
		print "Could not load word vectors from:", file_destination
		if file_destination == "word_vectors/glove.txt" or file_destination == "word_vectors/paragram.txt":
			print "Please unzip the provided glove/paragram vectors in the word_vectors directory.\n"
		return {}

	print len(word_dictionary), "vectors loaded from", file_destination			
	return normalise_word_vectors(word_dictionary)


def print_word_vectors(word_vectors, write_path):
	"""
	This method saves the counter-fitted word vectors in a text file. 
	"""
	print "Saving the counter-fitted word vectors to", write_path, "\n"
	with open(write_path, "wb") as f_write:
		for key in word_vectors:
			print >>f_write, key, " ".join(map(str, numpy.round(word_vectors[key], decimals=6))) 


def print_all_vectorst(word_vectors, all_vectors, write_path):
	# print all_vectors['the']
	# new_vector_space = all_vectors.copy()
	temp = all_vectors['the']
	print '*************' , len(all_vectors)
	for key, value in word_vectors.iteritems():
		all_vectors[key] = value
	# print temp - all_vectors['the']

	with open(write_path, "wb") as f_write:
		for key in all_vectors:
			if key == 'the':
				print temp - all_vectors[key]
				print key
			print >> f_write, key, " ".join(map(str, numpy.round(all_vectors[key], decimals=6)))
			

def normalise_word_vectors(word_vectors, norm=1.0):
	"""
	Normalizing the word vectors in the word_vectors dictionary.
	"""
	for word in word_vectors:
		word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
		word_vectors[word] = word_vectors[word] * norm
	return word_vectors


def load_constraints(constraints_filepath, vocabulary):
	"""
	This methods loads the constraints from the specified file, and returns a set with
	all constraints for which both of their constituent words are in the specified vocabulary.
	"""
	constraints_filepath.strip()
	constraints = set()
	with open(constraints_filepath, "r+") as f:
		for line in f:
			word_pair = line.split()
			if word_pair[0] in vocabulary and word_pair[1] in vocabulary and word_pair[0] != word_pair[1]:
				constraints |= {(word_pair[0], word_pair[1])}
				constraints |= {(word_pair[1], word_pair[0])}

	print constraints_filepath, "yielded", len(constraints), "constraints."

	return constraints


def distance(v1, v2, normalised_vectors = True):
	"""
	Returns the cosine distance between two vectors. 
	If the vectors are normalised, there is no need for the denominator, which is always one. 
	"""
	if normalised_vectors:
		return 1 - dot(v1, v2)
	else:
		return 1 - dot(v1, v2) / ( norm(v1) * norm(v2) )


def compute_vsp_pairs(word_vectors, vocabulary, rho=0.2):
	"""
	This method returns a dict with all word pairs that are closer that rho.
	To manage memory, it first computes the dot products of subsets, then recinstructs 
	the indices of the word vectors that are to be similar. 
	
	The pairs are mapped to the distance in the vector space. 	
	"""
	# first computing the pair of words that have Cosine similarity more than .8
	
	print "Pre-computing word pairs relevant for Vector Space Preservation (VSP). Rho =", rho

	vsp_pairs = {}

	threshold = 1 - rho 
	vocabulary = list(vocabulary)
	num_words = len(vocabulary)

	step_size = 1000 # size of word vectors to consider at each step. 
	vector_size = random.choice(word_vectors.values()).shape[0]

	# ranges of indices:
	list_of_ranges = []

	left_range_limit = 0
	while left_range_limit < num_words:
		curr_range = (left_range_limit, min(num_words, left_range_limit + step_size))
		list_of_ranges.append(curr_range)
		left_range_limit += step_size

	range_count = len(list_of_ranges)

	# In each word range, computing the word similarities. 
	for left_range in range(range_count):
		for right_range in range(left_range, range_count):

			# offsets the current word ranges:
			left_translation = list_of_ranges[left_range][0]
			right_translation = list_of_ranges[right_range][0]

			# copy the word vectors of the current word ranges:
			vectors_left = numpy.zeros((step_size, vector_size), dtype="float32")
			vectors_right = numpy.zeros((step_size, vector_size), dtype="float32")

			# two iterations as the two ranges need not be same length (implicit zero-padding):
			full_left_range = range(list_of_ranges[left_range][0], list_of_ranges[left_range][1])		
			full_right_range = range(list_of_ranges[right_range][0], list_of_ranges[right_range][1])
			
			for iter_idx in full_left_range:
				vectors_left[iter_idx - left_translation, :] = word_vectors[vocabulary[iter_idx]]

			for iter_idx in full_right_range:
				vectors_right[iter_idx - right_translation, :] = word_vectors[vocabulary[iter_idx]]

			# now compute the correlations between the two sets of word vectors: 
			dot_product = vectors_left.dot(vectors_right.T)

			# find the indices of those word pairs whose dot product is above the threshold:
			indices = numpy.where(dot_product >= threshold)

			num_pairs = indices[0].shape[0]
			left_indices = indices[0]
			right_indices = indices[1]
			
			for iter_idx in range(0, num_pairs):
				
				left_word = vocabulary[left_translation + left_indices[iter_idx]]
				right_word = vocabulary[right_translation + right_indices[iter_idx]]

				if left_word != right_word:
					# reconstruct the cosine distance and add word pair (both permutations):
					score = 1 - dot_product[left_indices[iter_idx], right_indices[iter_idx]]
					vsp_pairs[(left_word, right_word)] = score
					vsp_pairs[(right_word, left_word)] = score
		
	# print "There are", len(vsp_pairs), "VSP relations to enforce for rho =", rho, "\n"
	return vsp_pairs


def vector_partial_gradient(u, v, normalised_vectors=True):
	"""
	This function returns the gradient of cosine distance: \frac{ \partial dist(u,v)}{ \partial u}
	If they are both of norm 1 (we do full batch and we renormalise at every step), we can save some time.
	"""
	if normalised_vectors:
		gradient = u * dot(u,v)  - v 
	else:		
		norm_u = norm(u)
		norm_v = norm(v)
		nominator = u * dot(u,v) - v * numpy.power(norm_u, 2)
		denominator = norm_v * numpy.power(norm_u, 3)
		gradient = nominator / denominator

	return gradient


def one_step_SGD(word_vectors, synonym_pairs, antonym_pairs, vsp_pairs, current_experiment, kmulti):
	"""
	This method performs a step of SGD to optimise the counterfitting cost function.
	"""
	new_word_vectors = deepcopy(word_vectors)

	gradient_updates = {}
	update_count = {}
	oa_updates = {}
	vsp_updates = {}

	# AR term:
	for (word_i, word_j) in antonym_pairs:

		current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

		if current_distance < current_experiment.delta:
	
			gradient = vector_partial_gradient( new_word_vectors[word_i], new_word_vectors[word_j])
			gradient = gradient * current_experiment.hyper_k1 * kmulti

			if word_i in gradient_updates:
				gradient_updates[word_i] += gradient
				update_count[word_i] += 1
			else:
				gradient_updates[word_i] = gradient
				update_count[word_i] = 1

	# SA term:
	for (word_i, word_j) in synonym_pairs:

		current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

		if current_distance > current_experiment.gamma: 
		
			gradient = vector_partial_gradient(new_word_vectors[word_j], new_word_vectors[word_i])
			gradient = gradient * current_experiment.hyper_k2 * kmulti

			if word_j in gradient_updates:
				gradient_updates[word_j] -= gradient
				update_count[word_j] += 1
			else:
				gradient_updates[word_j] = -gradient
				update_count[word_j] = 1
	
	# VSP term:			
	for (word_i, word_j) in vsp_pairs:

		original_distance = vsp_pairs[(word_i, word_j)]
		new_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])
		
		if original_distance <= new_distance: 

			gradient = vector_partial_gradient(new_word_vectors[word_i], new_word_vectors[word_j]) 
			gradient = gradient * current_experiment.hyper_k3 

			if word_i in gradient_updates:
				gradient_updates[word_i] -= gradient
				update_count[word_i] += 1
			else:
				gradient_updates[word_i] = -gradient
				update_count[word_i] = 1

	for word in gradient_updates:
		# we've found that scaling the update term for each word helps with convergence speed. 
		update_term = gradient_updates[word] / (update_count[word]) 
		new_word_vectors[word] += update_term 
		
	return normalise_word_vectors(new_word_vectors)


def counter_fit(current_experiment,kmulti=1):
	"""
	This method repeatedly applies SGD steps to counter-fit word vectors to linguistic constraints. 
	"""
	word_vectors = current_experiment.pretrained_word_vectors
	vocabulary = current_experiment.vocabulary
	antonyms = current_experiment.antonyms
	synonyms = current_experiment.synonyms
	
	current_iteration = 0
	
	vsp_pairs = {}

	if current_experiment.hyper_k3 > 0.0: # if we need to compute the VSP terms.
 		vsp_pairs = compute_vsp_pairs(word_vectors, vocabulary, rho=current_experiment.rho)
	
	# Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
	for antonym_pair in antonyms:
		if antonym_pair in synonyms:
			synonyms.remove(antonym_pair)
		if antonym_pair in vsp_pairs:
			del vsp_pairs[antonym_pair]

	max_iter = 20
	print "\nAntonym pairs:", len(antonyms), "Synonym pairs:", len(synonyms), "VSP pairs:", len(vsp_pairs)
	print "Running the optimisation procedure for", max_iter, "SGD steps..."

	while current_iteration < max_iter:
		current_iteration += 1
		word_vectors = one_step_SGD(word_vectors, synonyms, antonyms, vsp_pairs, current_experiment,kmulti)

	return word_vectors

def load_all_word_vectors(file_destination):
	"""
	This method loads the word vectors from the supplied file destination. 
	It loads the dictionary of word vectors and prints its size and the vector dimensionality. 
	"""
	print "Loading all pretrained word vectors from", file_destination
	word_dictionary = {}

	try:
		with open(file_destination, "r") as f:
			for line in f:
				line = line.split(" ", 1)
				key = line[0].lower()
				word_dictionary[key] = numpy.fromstring(line[1], dtype="float32", sep=" ")
	except:
		print "Word vectors could not be loaded from:", file_destination
		if file_destination == "word_vectors/glove.txt" or file_destination == "word_vectors/paragram.txt":
			print "Please unzip the provided glove/paragram vectors in the word_vectors directory.\n"
		return {}

	print len(word_dictionary), "vectors loaded from", file_destination
	return normalise_word_vectors(word_dictionary)


def run_experiment(config_filepath):
	"""
	This method runs the counterfitting experiment, printing the SimLex-999 score of the initial
	vectors, then counter-fitting them using the supplied linguistic constraints. 
	We then print the SimLex-999 score of the final vectors, and save them to a .txt file in the 
	results directory.
	"""
	current_experiment = ExperimentRun(config_filepath)
	if not current_experiment.pretrained_word_vectors:
		return
	
	'''
	Set to 10 to consider values between 0.1 and 1 for k1 and k2.
	'''
	emotion_importance = 1
	for i in range(0, emotion_importance):
		print '########### emotion importance out of 10 = ', i
		transformed_word_vectors = counter_fit(current_experiment,kmulti=i+1)
		
		fname = "results/counter_fitted_vectors-"+str(i)+".txt"
		print_word_vectors(transformed_word_vectors, fname)

		newvects = load_all_word_vectors(current_experiment.pretrained_vectors_filepath)

		fname = "results/counter_fitted_vector_space-"+str(i)+".txt"

		print_all_vectorst(transformed_word_vectors, newvects, fname)


def main():
	"""
	The user can provide the location of the config file as an argument. 
	If no location is specified, the default config file (experiment_parameters.cfg) is used.
	"""
	try:
		config_filepath = sys.argv[1]
	except:
		print "\nUsing the default config file: experiment_parameters.cfg"
		config_filepath = "experiment_parameters.cfg"

	run_experiment(config_filepath)


if __name__=='__main__':
	main()
