# cifar CNN

from neon.backends import gen_backend
from neon.callbacks.callbacks import Callbacks
from neon.data import load_cifar10, load_mnist, ArrayIterator
from neon.initializers import Gaussian
from neon.layers import Affine, Conv, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import CrossEntropyMulti, Misclassification, Rectlin, Softmax
# Enables customisation flags including whether to use CPU or GPU
from neon.util.argparser import NeonArgparser
from archi import Architecture


be = gen_backend(backend='cpu', batch_size=128)

# doesn't actually do anything with the doc string!
parser = NeonArgparser(__doc__)

# Creates a "namespace" or backend which is then put into the original
# parser instantiation.
args = parser.parse_args()

# To train a deep network we need to specify the following:
# - dataset
(X_train, y_train), (X_test, y_test), nclass = load_cifar10()

# lshape tells the cnn what shape each row should be resized to since otherwise
# it's a 3 x 32 x 32 shape array.
train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(3, 32, 32))

test_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(3, 32, 32))

# - list of layers
arch = Architecture('AlexNet', nclass)

cnn = Model(layers=arch.layers)

# - cost function
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# - learning rule
optimizer = GradientDescentMomentum(arch.learning_rate, momentum_coef=arch.momentum)

# Progress bar for each epoch - what's an epoch again? by default 10 Crazy magic - don't even go here!
callbacks = Callbacks(cnn, eval_set=test_set, **args.callback_args)

# put everything together!
cnn.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

# Calculate test set results
results = cnn.get_outputs(test_set)

# work out the performance!
error = mlp.eval(test_set, metric=Misclassification())
