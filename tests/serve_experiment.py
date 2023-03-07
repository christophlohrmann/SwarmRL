import socket
import swarmrl.engine.real_experiment
import swarmrl.models.bechinger_models
import numpy as np

# create TCP socket for communication
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("localhost", 22009))
sock.listen(1)
print("listening on {}:{}".format(*("localhost", 22009)))
# wait for matlab to connect
connection, client_address = sock.accept()
print("Connected to {}:{}".format(*("localhost", 22009)))

experiment = swarmrl.engine.real_experiment.RealExperiment(connection)
perception_threshold = np.pi/2 * 82 / (np.pi**2 * 100)
force_model = swarmrl.models.bechinger_models.Lavergne2019(perception_threshold=0.05)

experiment.integrate(1000000000000000, force_model)

connection.close()