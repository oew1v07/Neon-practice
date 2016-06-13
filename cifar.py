# cifar CNN

from neon.backends import gen_backend
from neon.callbacks.callbacks import Callbacks
from neon.data import load_cifar10, load_mnist, ArrayIterator
from neon.initializers import Gaussian, Uniform
from neon.layers import Affine, Conv, GeneralizedCost, Pooling
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import CrossEntropyMulti, Misclassification, Rectlin, Softmax
# Enables customisation flags including whether to use CPU or GPU
from neon.util.argparser import NeonArgparser
from archi import Architecture
from joblib import dump


be = gen_backend(backend='cpu', batch_size=128)

# doesn't actually do anything with the doc string!
parser = NeonArgparser(__doc__)

# Creates a "namespace" or backend which is then put into the original
# parser instantiation.
args = parser.parse_args()

epochs = 10

# To train a deep network we need to specify the following:
# - dataset
(X_train, y_train), (X_test, y_test), nclass = load_cifar10()

# lshape tells the cnn what shape each row should be resized to since otherwise
# it's a 3 x 32 x 32 shape array.
train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(3, 32, 32))

test_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(3, 32, 32))

# - list of layers
arch = Architecture('AlexNet', nclass)
layers = arch.layers
learning_rate = arch.learning_rate
momentum = arch.momentum

# using the code provided by neon
# init_uni = Uniform(low=-0.1, high=0.1)
# layers = [Conv(fshape=(5,5,16), init=init_uni, activation=Rectlin()),
#           Pooling(fshape=2, strides=2),
#           Conv(fshape=(5,5,32), init=init_uni, activation=Rectlin()),
#           Pooling(fshape=2, strides=2),
#           Affine(nout=500, init=init_uni, activation=Rectlin()),
#           Affine(nout=10, init=init_uni, activation=Softmax())]
# learning_rate = 0.005
# momentum = 0.9

cnn = Model(layers=layers)

# - cost function
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# - learning rule
optimizer = GradientDescentMomentum(learning_rate, momentum_coef=momentum)

# Progress bar for each epoch - what's an epoch again? by default 10 Crazy magic - don't even go here!
callbacks = Callbacks(cnn, eval_set=test_set, **args.callback_args)

# put everything together!
cnn.fit(train_set, optimizer=optimizer, num_epochs=epochs, cost=cost, callbacks=callbacks)

# # Calculate test set results
# results = cnn.get_outputs(test_set)

# dump(cnn, "cnn_0_005.jbl")

# # work out the performance!
# error = cnn.eval(test_set, metric=Misclassification())
