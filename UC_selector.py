import numpy as np

def probabilistic_uncertainty(probs, n):
    """Perform uncertainty sampling using least confidence.

    An excellent discussion of active learning strategies including a
    comparison of three different uncertainty measures: least confidence,
    margin sampling and entropy (all of which are equivalent in binary
    classification) can be found in Burr Settles' literature review:

    http://burrsettles.com/pub/settles.activelearning.pdf

    The process for selecting the objects is as follows:

        1. Consider 'uncertainty' only (1 - the highest class probability).
        2. Argsort and reverse to sort from least to most certain.
        3. Take the n smallest (most uncertain).

    Args:
        probs: The list of probabilities to use as metrics.
        n: The number of samples to mark as 'most uncertain'.

    Returns:
        list: The indexes corresponding to the 'most uncertain' samples.

    """
    uncertainty = np.array([1 - np.max(x) for x in probs])
    indexes = np.argsort(uncertainty)[::-1]
    return indexes[:n]

