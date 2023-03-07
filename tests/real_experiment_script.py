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


force_model_sym = simon_models.rotate_rod_border_schmell_symmetric(
    act_force=10.0,
    act_torque=10.0,
    n_type=[1000,100],
    rod_schmell_part_id=[10000,10059],
    rod_center_part_id=30,
    rod_particle_type=1,
    rod_thickness=100/60/2, #rodlength/particle number
    radius_colloid=3.0,
    force_team_spirit_fac=0,
    rod_break_ang_vel= 42,
    rod_break= False,
)

force_model_zickzack = simon_models.zickzack_pointfind(
    act_force=10.0,
    act_torque=10.0,
    n_type=[1000,100],
    center_point=[500,500],
    phase_len=[30, 420, 420, 64],
    diffusion_coeff=1.4,
    t_slice = 10 , # in seconds time between calculation of the tasks 
    steer_speed = 0.8 * np.pi / 180,  # rad per t_slice
    len_run = 20 ,# mu
    run_speed = 0.1 , #mu/t_slice
    zick_angle = 45,
    experiment_engine=False
)


experiment.integrate(100000000000, force_model_no_int)
experiment.finalize()

connection.close()