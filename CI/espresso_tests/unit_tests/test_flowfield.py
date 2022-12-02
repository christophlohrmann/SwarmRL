import tempfile
import unittest as ut

import numpy as np
import pint

from swarmrl.engine import espresso
from swarmrl.models import dummy_models


class FlowFieldTest(ut.TestCase):
    def test_class(self):
        """
        we want to eliminate friction from the brownian thermostat because we
        already get friction from the field. Therefore, we decrease both
        viscosity and temperature. The noise is proportional to the quotient
        and not affected, whereas the friction gets reduced.
        """
        friction_scaling_factor = 1
        ureg = pint.UnitRegistry()
        water_visc = ureg.Quantity(8.9e-4, "pascal * second")
        room_temp = ureg.Quantity(300, "kelvin")

        params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=water_visc * friction_scaling_factor,
            WCA_epsilon=0.1 * ureg.Quantity(300, "kelvin") * ureg.boltzmann_constant,
            temperature=room_temp * friction_scaling_factor,
            box_length=ureg.Quantity(50, "micrometer"),
            time_step=ureg.Quantity(0.001, "second"),
            time_slice=ureg.Quantity(0.1, "second"),
            write_interval=ureg.Quantity(0.1, "second"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = espresso.EspressoMD(
                params, n_dims=2, out_folder=temp_dir, write_chunk_size=1
            )
            self.assertListEqual(runner.colloids, [])

            n_colloids = 50
            coll_rad = ureg.Quantity(1.0, "micrometer")
            runner.add_colloids(
                n_colloids,
                coll_rad,
                ureg.Quantity(np.array([25, 25, 0]), "micrometer"),
                ureg.Quantity(25, "micrometer"),
                type_colloid=0,
            )

            flow_field = ureg.Quantity(np.array([1, 2, 3]), "micrometer/second") * 1e-16
            # manually calculate friction coefficient
            fric = 6 * np.pi * coll_rad * water_visc

            runner.add_constant_flowfield(flow_field, friction_coefficient=fric)

            pos_before = runner.get_particle_data()["Unwrapped_Positions"]
            no_force = dummy_models.ConstForce(force=0)
            runner.integrate(100, no_force)
            pos_after = runner.get_particle_data()["Unwrapped_Positions"]

            # on average, the colloids should have moved with the flowfield
            print(pos_after - pos_before)


if __name__ == "__main__":
    ut.main()
