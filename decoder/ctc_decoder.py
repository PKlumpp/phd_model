from typing import Tuple
import numpy as np
from fast_ctc_decode import viterbi_search

def decode_lattice(
    lattice: np.ndarray,
    enc_feats: np.ndarray = None,
    cnn_feats: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Blank index must be 0
    Input lattice is expected in the form of (T, S), without batch dimension
    Outputs state sequence (phones), along with encoder features, cnn features and softmax probability of emitted symbol
    """
    _, path = viterbi_search(lattice, alphabet=np.arange(lattice.shape[-1]))
    probs = lattice[path, :]
    states = np.argmax(probs, axis=1)
    probs = probs[np.arange(len(states)), states]
    enc = None
    if enc_feats is not None:
        enc = enc_feats[path]
    cnn = None
    if cnn_feats is not None:
        cnn = cnn_feats[path]
    return states, enc, cnn, probs
