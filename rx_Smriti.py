import socket
import pickle
import numpy as np

UDP_IP = "127.0.0.1"
C_RX_UDP_PORT=5007
RX_C_UDP_Port=5001
CH_RX_UDP_PORT=5015

MESSAGE_Rx_Pilot = b"Receive Pilot"
MESSAGE_SEND_RESULTS=b"Send Results"
Message_Signal = b"Message Data"

sockcrx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockcrx.bind((UDP_IP, C_RX_UDP_PORT))
sockchrx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockchrx.bind((UDP_IP, CH_RX_UDP_PORT))

def add_ktb_noise(signal, B_hz, T):
    k = 1.38 * 10**(-23)
    noise_power = k*B_hz*T
    sigma = np.sqrt(noise_power)
    noise = (1000 * sigma / (np.sqrt(2)) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)))
    # np.random.randn(*signal.shape) expression , generates an array of random numbers with the same shape as the signal array.
    return signal + noise, np.abs(noise)**2

y_list=[]
Fs = 1e6 # sampling Frequency
sps =10 # samples per symbols (per I, Q samples)
Ts = (sps / Fs) # time gap between two samples
B_hz = 1/(2*Ts) # BAndWidth

first_noise = None  # To store the noise once

while True:
    data, addr = sockcrx.recvfrom(1024) # buffer size is 1024 bytes
    if data== MESSAGE_Rx_Pilot:
        data1, addr1 = sockchrx.recvfrom(65535) # buffer size is 1024 bytes
        y1 = np.array(pickle.loads(data1))
        y, noise_power = add_ktb_noise(y1,B_hz ,300)
        if first_noise is None:
            first_noise = noise_power   # Store noise only once
        print("Received the signal ",y1, "from Channel")
        print("=================================================================================")
        y_list.append(y) # if H1 and H2 change, y_list should be refreshed
    if data==MESSAGE_SEND_RESULTS:
        final_list = [first_noise] + y_list  # noise first, then all y[n]
        serial_list_y = pickle.dumps(final_list)
        sockcrx.sendto(serial_list_y, (UDP_IP, RX_C_UDP_Port))
        print("Sent the signal ",y_list, "to Controller")
        y_list=[]

