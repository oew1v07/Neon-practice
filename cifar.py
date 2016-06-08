# cifar CNN

from neon.callbacks.callbacks import Callbacks
from neon.data import load_cifar10, load_mnist, ArrayIterator
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
(X_train, y_train), (X_test, y_test), nclass = load_cifar10()