import numpy as np
import simon_models
import socket
import swarmrl.engine.real_experiment


# Experiment communication
# create TCP socket for communication
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("localhost", 22009))
sock.listen(1)
print("listening on {}:{}".format(*("localhost", 22009)))
# wait for matlab to connect
connection, client_address = sock.accept()
print("Connected to {}:{}".format(*("localhost", 22009)))




experiment = swarmrl.engine.real_experiment.RealExperiment(connection)
experiment.setup_simulation()

radii= [3.0]*10000 +[0.1]*100


force_model_no_int = simon_models.rotate_rod_vision_cone(
    data_folder="None",
    act_force=10.0,
    act_torque=10.0,
    n_type=[1000,100],#both numbers need to be larger than the possible amount in the experiment
    rod_particle_type=1, # colloid type is assumed to be 0 can change with act_on_type =[2,3]
    radius_vector=radii,
    detection_radius_position=10000,
    vision_half_angle=1.4,
    phase_len=[42,42,42,42],
    experiment_engine=True
)

'''
force_model_int = simon_models.rotate_rod_vision_cone_interaction(
    data_folder="None",
    act_force=10.0,
    act_torque=10.0,
    n_type=[1000,100],#both numbers need to be larger than the possible amount in the experiment
    rod_particle_type=1, # colloid type is assumed to be 0 can change with act_on_type =[2,3]
    radius_vector=radii,
    detection_radius_position=10000,
    vision_half_angle=1.4,
    phase_len=[42,42,42,42],
    experiment_engine=True
)
'''


experiment.integrate(100000000000, force_model_no_int)
experiment.finalize()

connection.close()