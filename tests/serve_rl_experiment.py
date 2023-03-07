"""
Test the RL deployment in experiment.
"""

import socket
import swarmrl.engine.real_experiment
from swarmrl.models import MLModel
import numpy as np
from swarmrl.models.interaction_model import Action
import optax
import swarmrl as srl
import flax.linen as nn

# Experiment communication
# create TCP socket for communication
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("localhost", 22009))
sock.listen(1)
print("listening on {}:{}".format(*("localhost", 22009)))
# wait for matlab to connect
connection, client_address = sock.accept()
print("Connected to {}:{}".format(*("localhost", 22009)))

# SwarmRL operation
experiment = swarmrl.engine.real_experiment.RealExperiment(connection)

# Define the actions
translate = Action(force=10.0)
rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0]))
rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
do_nothing = Action()

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
    source=np.array([100.0, 100.0, 0.0]),
    decay_function=scale_function,
    reward_scale_factor=10,
    box_size=np.array([1000.0, 1000.0, 1000]),
)
observable = task.init_task()

colloids = experiment.receive_colloids()
observable.initialize(colloids) 

# dummy send
colloid_idx = np.arange(0, len(colloids), 1).reshape(-1, 1)
colloid_action = np.ones_like(colloid_idx).reshape(-1, 1)
send_req = np.hstack((colloid_idx, colloid_action))
print(send_req)
experiment.send_actions(send_req)

deploy_model = srl.networks.FlaxModel(
    flax_model=ActorNet(),
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1,),
    sampling_strategy=sampling_strategy,
    deployment_mode=False
)
deploy_model.restore_model_state(directory="/home/veit_lab/Git/swarmrl_experiment/new/SwarmRL/tests/", filename="ActorModel_0")

force_fn = srl.models.ml_model.MLModel(
    models={"0": deploy_model},
    observables={"0": observable},
    tasks={"0": task},
    actions={"0": actions},
    record_traj=False  # Only used during training, turn it off here.
)

experiment.integrate(1000000000000000, force_fn)

connection.close()