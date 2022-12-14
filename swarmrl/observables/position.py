"""
Position observable computer.
"""
from abc import ABC

import jax.numpy as np
import numpy as onp

from .observable import Observable


class PositionObservable(Observable, ABC):
    """
    Position in box observable.
    """

    _observable_shape = (3,)

    def __init__(self, box_length: np.ndarray):
        """
        Constructor for the observable.

        Parameters
        ----------
        box_length : np.ndarray
                Length of the box with which to normalize.
        """
        self.box_length = box_length

    def compute_observable(self, colloid: object, other_colloids: list):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.

        Returns
        -------

        """
        data = onp.copy(colloid.pos)

        return np.array(data) / self.box_length
