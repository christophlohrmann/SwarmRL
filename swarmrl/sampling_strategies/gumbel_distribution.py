"""
Module for the Gumbel distribution.
"""
from abc import ABC

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.sampling_strategies.sampling_stratgey import SamplingStrategy


class GumbelDistribution(SamplingStrategy, ABC):
    """
    Class for the Gumbel distribution.
    """

    def compute_entropy(self, probabilities: np.ndarray):
        """
        Compute the entropy of the distribution.

        Parameters
        ----------
        probabilities : np.ndarray
                Array of probabilities on which to compute the entropy.

        Returns
        -------
        entropy : float
                Returns the shannon entropy of the distribution.

        """
        entropy_val = np.sum(-1 * (probabilities * np.log(probabilities)), axis=-1)
        entropy_val = np.mean(entropy_val)

        return entropy_val

    def __call__(self, logits: np.ndarray, **kwargs):
        """
        Sample from the distribution.

        Parameters
        ----------
        logits : np.ndarray
                Logits from the model to use in the computation.

        Returns
        -------
        sample : int
                Index of the selected option in the distribution.

        Notes
        -----
        See https://arxiv.org/abs/1611.01144 for more information.
        """
        rng = jax.random.PRNGKey(onp.random.randint(0, 1236534623))
        length = len(logits)
        noise = jax.random.uniform(rng, shape=(length,))

        index = np.argmax(logits - np.log(-np.log(noise)))
        return index
