# Specific architectures, the specifics of this were gained from 
# a mixture of the Krizhevsky et al paper and 
# http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf

from neon.initializers import Constant, Gaussian, Uniform
from neon.layers import Affine, Conv, Dropout, LRN, Pooling
from neon.transforms import Rectlin, Softmax

class Architecture(object):
	# We might want to turn this into a class which takes the type as
	# an argument either 'AlexNet' or 'GoogLeNet' which will allow the
	# object to have attributes of learning rate depending on what
	# each paper used
	def __init__(self, arch_type='AlexNet', nclass=10, dataset='cifar', learning_rate=None):
		self.arch_type = arch_type
		if arch_type == 'AlexNet':
			if arch_type == "cifar":
				self.example(nclass=nclass, dataset=dataset, learning_rate = learning_rate)
			else:
				self.AlexNet(nclass=nclass, dataset=dataset, learning_rate = learning_rate)
		else:
			print("There's no such {} architecture yet".format(self.arch_type))

	def AlexNet(self, nclass=10, dataset='cifar', learning_rate=None):
		# We set a number of the attributes
		self.dataset = dataset
		# Dropout rate is the percentage of dropout in the CNN (i.e. 0.3 is 30%)
		# Initial dropout is the dropout within the first layer.
		# In this model dropout is only used in the first two layers
		self.dropout_rate = 0.5

		# This is eta or the weight decay
		if learning_rate is None:
			self.learning_rate = 0.0005
		else:
			self.learning_rate = learning_rate

		self.momentum = 0.9 # this never varies!

		self.batch_size = 128

		self.init_type = 'Gaussian(0, 0.01)'

		self.bias = 1

		# Biases are initialised in 2nd, 4th and 5th conv layers, with constant 1
		self._biases = [None, 1, None, 1, 1]

		# Parameters for the Local response normalisation
		self.depth = 5 # number of adjacent pixels, however they have used -1 to neaten the maths
		# ascale/N  is equivalent to alpha/k in original equation. alpha = 10^-4, k = 2
		# However we have no control over N so I have made it equal to the alpha value.
		self.ascale = 0.0001
		self.bpower = 0.75 # also known as beta

		self._LRNs = [1, 1, None, None, None]

		# which layers have pooling
		self._pools = ['max', 'max', None, None, 'max']

		self.full_num = 4096

		# So the first five layers of AlexNet are convolutional
		# and the last 3 are fully connected. The final one is a softmax
		# with the number of classes
		layers = []

		# Need to specify the filter size for each convolution and the number
		# fshape = (height, width, num) so fshape is meant to have three 
		# dimensions and then a number of kernels - not sure how this is 
		# implemented. Is the third dimension inferred from the shape of the
		# input? The documentation definitely says (height, width, num_filters)
		# the commented portions are the actual filter size.

		fshape = []

			# This is for the image inputs being 224. Not suitable for cifar
		fshape.append((11, 11, 96)) # or (11, 11, 3)
		fshape.append((5, 5, 256)) # or (5, 5, 48)
		fshape.append((3, 3, 384)) # or (3, 3, 256)
		fshape.append((3, 3, 384)) # or (3, 3, 192)
		fshape.append((3, 3, 256)) # or (3, 3, 192)

		# Each max pooling uses a 2x2 window and CNN uses stride of 4.

		pool_shape = 2

		strides = 4

		# This is used in the Krizhevsky paper

		init_norm = Gaussian(loc=0.0, scale=0.01)

		for i, shape in enumerate(fshape):

			if self._biases[i]:
				layers.append(Conv(shape, strides=strides, init=init_norm, activation=Rectlin(), bias=Constant(self._biases[i])))
			else:
				layers.append(Conv(shape, strides=strides, init=init_norm, activation=Rectlin()))
			

			# There is no need for input normalization with ReLUs but does decrease error
			# in this the defaults are used (below) not sure what values were used.
			# alpha = 1.0
			# beta = 1.0
			# ascale = 1.0
			# bpower = 1.0
			if self._LRNs[i]:
				layers.append(LRN(depth=self.depth, ascale=self.ascale, bpower=self.bpower))

			if self._pools[i]:
				print("Using pooling")
				# overlapping scheme of pooling decreases error, stride = None at the moment
				layers.append(Pooling(pool_shape, op=self._pools[i]))	

		layers.append(Affine(nout=self.full_num, init=init_norm, activation=Rectlin()))

		# We need dropout in these two layers
		layers.append(Dropout(keep=self.dropout_rate))

		layers.append(Affine(nout=self.full_num, init=init_norm, activation=Rectlin()))
		# We need dropout in these two layers
		layers.append(Dropout(keep=self.dropout_rate))

		layers.append(Affine(nout=nclass, init=init_norm, activation=Softmax()))

		self.layers = layers

	def example(self, nclass=10, dataset='cifar', learning_rate=None):
		# We set a number of the attributes
		self.dataset = dataset
		# Dropout rate is the percentage of dropout in the CNN (i.e. 0.3 is 30%)
		# Initial dropout is the dropout within the first layer.
		# In this model dropout is only used in the first two layers
		self.dropout_rate = 0.5

		# This is eta or the weight decay

		if learning_rate is None:
			self.learning_rate = 0.005
		else:
			self.learning_rate = learning_rate

		self.momentum = 0.9 # this never varies!

		self.batch_size = 128

		self.init_type = 'Uniform(-0.1, 0.1)'

		self.bias = None

		# Biases are not initialised as yet but code is here to change that
		self._biases = [None, None, None, None, None]

		# Parameters for the Local response normalisation
		self.depth = 5 # number of adjacent pixels, however they have used -1 to neaten the maths
		# ascale/N  is equivalent to alpha/k in original equation. alpha = 10^-4, k = 2
		# However we have no control over N so I have made it equal to the alpha value.
		self.ascale = 0.0001
		self.bpower = 0.75 # also known as beta

		# No LRN at the moment but 
		self._LRNs = [None, None, None, None, None]

		# which layers have pooling
		self._pools = ['max', 'max', None, None, 'max']

		if dataset == 'cifar':
			self.full_num = 500
		else:
			self.full_num = 4096

		# So the first five layers of AlexNet are convolutional
		# and the last 3 are fully connected. The final one is a softmax
		# with the number of classes
		layers = []

		# Need to specify the filter size for each convolution and the number
		# fshape = (height, width, num) so fshape is meant to have three 
		# dimensions and then a number of kernels - not sure how this is 
		# implemented. Is the third dimension inferred from the shape of the
		# input? The documentation definitely says (height, width, num_filters)
		# the commented portions are the actual filter size.

		fshape = []
		if dataset == 'cifar':
			fshape.append((5, 5, 16))
			fshape.append((5, 5, 32))
		else:
			# This is for the image inputs being 224. Not suitable for cifar
			fshape.append((11, 11, 96)) # or (11, 11, 3)
			fshape.append((5, 5, 256)) # or (5, 5, 48)
			fshape.append((3, 3, 384)) # or (3, 3, 256)
			fshape.append((3, 3, 384)) # or (3, 3, 192)
			fshape.append((3, 3, 256)) # or (3, 3, 192)

		# Each max pooling uses a 2x2 window.

		pool_shape = 2

		strides = 2

		# This is used in the Krizhevsky paper
		if dataset == "cifar":
			init_norm = Uniform(low=-0.1, high=0.1)
		else:
			init_norm = Gaussian(loc=0.0, scale=0.01)

		for i, shape in enumerate(fshape):

			if self._biases[i]:
				layers.append(Conv(shape, strides=strides, init=init_norm, activation=Rectlin(), bias=Constant(self._biases[i])))
			else:
				layers.append(Conv(shape, strides=strides, init=init_norm, activation=Rectlin()))
			

			# There is no need for input normalization with ReLUs but does decrease error
			# in this the defaults are used (below) not sure what values were used.
			# alpha = 1.0
			# beta = 1.0
			# ascale = 1.0
			# bpower = 1.0
			if self._LRNs[i]:
				layers.append(LRN(depth=self.depth, ascale=self.ascale, bpower=self.bpower))

			if self._pools[i]:
				print("Using pooling")
				# overlapping scheme of pooling decreases error, stride = None at the moment
				layers.append(Pooling(pool_shape, op=self._pools[i]))	

		layers.append(Affine(nout=self.full_num, init=init_norm, activation=Rectlin()))

		if dataset != 'cifar':
			# We need dropout in these two layers
			layers.append(Dropout(keep=self.dropout_rate))

			layers.append(Affine(nout=self.full_num, init=init_norm, activation=Rectlin()))
			# We need dropout in these two layers
			layers.append(Dropout(keep=self.dropout_rate))

		layers.append(Affine(nout=nclass, init=init_norm, activation=Softmax()))

		self.layers = layers

