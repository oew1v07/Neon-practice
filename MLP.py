# MLP example

from neon.callbacks.callbacks import Callbacks
from neon.data import load_mnist, ArrayIterator
from neon.initializers import Gaussian
from neon.layers import Affine, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import CrossEntropyMulti, Misclassification, Rectlin, Softmax
# Enables customisation flags including whether to use CPU or GPU
from neon.util.argparser import NeonArgparser

# doesn't actually do anything with the doc string!
parser = NeonArgparser(__doc__)

# Creates a "namespace" or backend which is then put into the original
# parser instantiation.
args = parser.parse_args()


# They have thoughfully provided a function to load the data we want
(X_train, y_train), (X_test, y_test), nclass = load_mnist()

# Everything in neon needs an iterator to go through the examples within 
# an array. This needs to be done for both training and test set.
# This assumes that the data are laid out in an array the shape of
# (num_example, num_features)
# more examples http://neon.nervanasys.com/docs/latest/loading_data.html
train_set = ArrayIterator(X_train, y_train, nclass=nclass)

test_set = ArrayIterator(X_test, y_test, nclass=nclass)

# To train a deep network we need to specify the following:
# - dataset
# - list of layers
# - cost function
# - learning rule
# - Initialisation of weights (generally gaussian loc=mean, scale = std)

init_norm = Gaussian(loc=0.0, scale=0.01)

# Creating our architecture. Now we can play with this but also
# start off with AlexNet which can be pre-loaded like so:
# RELU is referred to as Rectlin.
# fully connected = Affine
# Softmax is used to make the final layer sum to one (corresponding to the
# number of classes) - it's a type of activation

# These components are appended to a list as follows
# [Fully connected layers of x hidden units (called nout), Fully connected final layer]

layers = []
layers.append(Affine(nout=100, init=init_norm,activation=Rectlin()))
layers.append(Affine(nout=10, init=init_norm,activation=Softmax()))


# we now construct the model
mlp = Model(layers=layers)

# We define our cost function - need to work out which is best
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# Learning rules (ie what kind of gradient descent, stochastic here)
# learning rate is 0.1 here, momentum coefficient (eta) is 0.9

optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

# Progress bar for each epoch - what's an epoch again? by default 10 Crazy magic - don't even go here!

callbacks = Callbacks(mlp, eval_set=test_set, **args.callback_args)

# put everything together!

mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost,
        callbacks=callbacks)

# Calculate test set results
results = mlp.get_outputs(test_set)

# work out the performance!
error = mlp.eval(test_set, metric=Misclassification())
print('Misclassification error rate is {}'.format(error))