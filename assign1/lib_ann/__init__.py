try:
    # fortran implementation is faster 30%.
    # Therefore, use fortran one
    from .ann_f import ann_for
    softmax = ann_for.softmax
except ImportError:
    print("Warning: Unable to load lib_ann.ann_f."
          " Use softmax python implementation.")
    softmax = ann._softmax_py

from . import ann
from .ann import evaluate_classifier
from .ann import compute_cost
from .ann import compute_accuracy
from .ann import compute_gradients
from .ann import evaluate_classifier

