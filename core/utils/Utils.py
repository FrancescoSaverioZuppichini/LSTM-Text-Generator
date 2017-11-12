import numpy as np

def sample_prob_picker_from_best(distribution, n=2):
    """
    Select n top best probability in a distribution
    :param n: Number of best possible candidates
    :return: 1D vector with the selected item
    """
    p = np.squeeze(distribution)
    p[np.argsort(p)[:-n]] = 0
    p = p / np.sum(p)

    return np.array([np.random.choice(distribution.shape[-1], 1, p=p)[0]])




