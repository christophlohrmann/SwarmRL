
"""
Run an RL agent to find the center of a box.
"""
import copy
import tempfile
import unittest as ut


import numpy as np

import pint
import flax.linen as nn
import swarmrl as srl
import swarmrl.engine.espresso as espresso
from swarmrl.utils import utils
from privat_utils import logger as loggbook
from swarmrl.models.interaction_model import Action
import optax



"""
Run the simulation.

Returns
-------

"""
loglevel_terminal = "info"
seed = 42
outfolder = "."

# manually turn on or off, cannot be checked in a test case
logger = utils.setup_swarmrl_logger(
    f"{outfolder}/deployment.log",
    loglevel_terminal=loglevel_terminal,
)
logger.info("Starting simulation setup")

ureg = pint.UnitRegistry()
md_params = espresso.MDParams(
    ureg=ureg,
    fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
    WCA_epsilon=ureg.Quantity(297.0, "kelvin") * ureg.boltzmann_constant,
    temperature=ureg.Quantity(300, "kelvin"),
    box_length=ureg.Quantity(1000, "micrometer"),
    time_slice=ureg.Quantity(0.5, "second"),  # model timestep
    time_step=ureg.Quantity(0.5, "second") / 10,  # integrator timestep
    write_interval=ureg.Quantity(2, "second"),
)

# parameters needed for bechinger_models.Baeuerle2020
model_params = {
    "target_vel_SI": ureg.Quantity(0.5, "micrometer / second"),
    "target_ang_vel_SI": ureg.Quantity(4 * np.pi / 180, "1/second"),
    "vision_half_angle": np.pi,
    "detection_radius_position_SI": ureg.Quantity(np.inf, "meter"),
    "detection_radius_orientation_SI": ureg.Quantity(25, "micrometer"),
    "angular_deviation": 67.5 * np.pi / 180,
}

run_params = {
    "n_colloids": 20,
    "sim_duration": ureg.Quantity(3, "minute"),
    "seed": seed,
}

system_runner = srl.espresso.EspressoMD(
    md_params=md_params,
    n_dims=2,
    seed=run_params["seed"],
    out_folder=outfolder,
    write_chunk_size=100,
)

coll_type = 0
system_runner.add_colloids(
    run_params["n_colloids"],
    ureg.Quantity(2.14, "micrometer"),
    ureg.Quantity(np.array([1000/4, 1000/4, 0]), "micrometer"),
    ureg.Quantity(0.2 * 1000, "micrometer"),
    type_colloid=coll_type,
)


# system_runner.add_confining_walls(wall_type = 100)

md_params_without_ureg = copy.deepcopy(md_params)
md_params_without_ureg.ureg = None

# Define the force model.

translate = Action(force=10.0)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0]))
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
do_nothing = Action()

# Define the loss model
actions = {
    "RotateClockwise": rotate_clockwise,
    "Translate": translate,
    "RotateCounterClockwise": rotate_counter_clockwise,
    "DoNothing": do_nothing,
}


class ActorNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        return x


def scale_function(distance: float):
    """
    Scaling function for the task
    """
    return 1 - distance


sampling_strategy = srl.sampling_strategies.GumbelDistribution()

task = srl.tasks.searching.GradientSensing(
    source=np.array([500.0, 500.0, 0.0]),
    decay_function=scale_function,
    reward_scale_factor=10,
    box_size=np.array([1000.0, 1000.0, 1000]),
)
observable = task.init_task()
observable.initialize(system_runner.colloids)

deploy_model = srl.networks.FlaxModel(
    flax_model=ActorNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1,),
    sampling_strategy=sampling_strategy,
    deployment_mode=False
)
deploy_model.restore_model_state(directory=".", filename="ActorModel_0")

force_fn = srl.models.ml_model.MLModel(
    models={"0": deploy_model},
    observables={"0": observable},
    tasks={"0": task},
    actions={"0": actions},
    record_traj=False  # Only used during training, turn it off here.
)

system_runner.integrate(2500, force_fn)
