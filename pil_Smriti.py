import socket
import pickle
import numpy as np

UDP_IP = "127.0.0.1"
PILOT_UDP_PORT = 5005
DATA_CHANNEL_UDP_PORT = 5010


MESSAGE_PILOT = b"Transmit Pilot"
No_of_I_Q_samples = 10 # samples per pilot
sockcc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockcc.bind((UDP_IP, PILOT_UDP_PORT))

sockpc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
while True:
    data, addr = sockcc.recvfrom(1024)  # buffer size is 1024 bytes
    received_list = pickle.loads(data)
    # config_number = received_list[1] # should not be informed to pilot generator
    pilot_power = received_list[1]
    pilot = np.sqrt(pilot_power) * np.ones((No_of_I_Q_samples,), dtype= np.complex64)
    print("Pilot sent: %s" % pilot)
    if received_list[0] == MESSAGE_PILOT:
        list_pilot = [MESSAGE_PILOT, pilot]
        list_pilot = pickle.dumps(list_pilot)
        sockpc.sendto(list_pilot, (UDP_IP, DATA_CHANNEL_UDP_PORT))



