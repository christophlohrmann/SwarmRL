"""
Test the harmonic trap potential.
"""
import unittest as ut

import jax.numpy as jnp
import numpy as np

from swarmrl.models.harmonic_trap import HarmonicTrap


class TestHarmonicTrapMethods(ut.TestCase):
    """
    Test the harmonic trap potential.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare the class.

        Returns
        -------

        """
        cls.model = HarmonicTrap(stiffness=10, center=np.array([0.0, 0.0, 0.0]))

        cls.colloids = [[0.0, 0.0, 0.0], [1.0, 3, 7.9], [-3.6, 3.2, -0.1]]

    def test_compute_force_simple(self):
        """
        Test the compute force method.

        Compute the forces on 3 particles in different locations but with a center at
        (0.0, 0.0, 0.0)

        Returns
        -------
        Will assert whether or not the forces are correct.
        """
        self.model.center = np.array([0.0, 0.0, 0.0])
        actual = np.array([[-0.0, -0.0, -0.0], [-10, -30, -79], [36, -32, 1.0]])
        prediction = self.model.compute_force(jnp.array(self.colloids))
        np.testing.assert_array_equal(prediction, actual)

    def test_compute_force_shifted(self):
        """
        Compute the forces as above with a shifted center.

        Returns
        -------
        Will assert whether or not the forces are correct.
        """
        self.model.center = np.array([1.0, 1.0, 1.0])

        actual = np.array([[10.0, 10.0, 10.0], [-0.0, -20.0, -69], [46.0, -22.0, 11.0]])
        prediction = self.model.compute_force(jnp.array(self.colloids))
        np.testing.assert_array_equal(prediction, actual)

    def test_compute_force_shifted_centers(self):
        """
        Compute the forces as above with a shifted center for each particle.

        Returns
        -------
        Will assert whether or not the forces are correct.
        """
        self.model.center = np.array(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]
        )

        actual = np.array([[10.0, 10.0, 10.0], [-10, -30, -79], [46.0, -42.0, 1.0]])
        prediction = self.model.compute_force(jnp.array(self.colloids))
        np.testing.assert_array_equal(prediction, actual)


if __name__ == "__main__":
    ut.main()
