import numpy as np
import matplotlib.pyplot as plt
import socket
import pickle

# === Signal Generation ===
def bits_to_symbol(bits):
    mapping = {
        1: 0.1 + 1j*0.1,
        0: -0.1 - 1j*0.1
    }
    sym = np.array([mapping[b] for b in bits], dtype=complex)
    return sym

def add_header(header_size, data):
    h_bits = np.ones(header_size, dtype=int)
    # known = bits_to_symbol(h_bits) * 5
    known = bits_to_symbol(h_bits)
    prep_data = np.concatenate((known, data))
    return prep_data

def message_final(data_size, sps, header_size):
    bitstream = [1]*5 + [0]*5 + [1]*5 + [0]*5# data example
    symbols = bits_to_symbol(bitstream)
    data_f = add_header(header_size, symbols)

    pulse = np.ones(sps)
    x = np.zeros(len(data_f) * sps, dtype=complex)
    x[::sps] = data_f
    tx = np.convolve(pulse, x)
    tmp = np.zeros(len(tx) * 2, dtype=np.float32)
    tmp[::2] = tx.real
    tmp[1::2] = tx.imag
    tx_c128=tx.real + 1j * tx.imag
    tx_c64= tx_c128.astype(np.complex64)
    return tx_c64

# === Generate Baseband Signal ===
data_size = 20
header_size = 10
Fs = 1e6 # sampling Frequency
sps =10 # samples per symbolS
Ts = (sps / Fs) # time gap between two samples
B_hz = 1/(2*Ts) # BAndWidth

signal = message_final(data_size, sps, header_size)

# === UDP Socket Sending ===
UDP_IP = "127.0.0.1"
CTRL_PORT = 5013
BPSK_CHANNEL_UDP_PORT = 5020
MESSAGE_Data = b"Transmit data"
Message_Transmission = b"Message Transmitted"

sockcc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockcc.bind((UDP_IP, CTRL_PORT))

sockpc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Transmitter is running... waiting for trigger")

while True:
    data, addr = sockcc.recvfrom(1024)
    received_list = pickle.loads(data)

    if received_list[0] == MESSAGE_Data:
        print("Trigger received, sending signal...")
        msg_power = received_list[1]
        tx_scaled = np.sqrt(msg_power) * signal

        packet = [Message_Transmission, tx_scaled, sps]
        packet_serialized = pickle.dumps(packet)
        sockpc.sendto(packet_serialized, (UDP_IP, BPSK_CHANNEL_UDP_PORT))
