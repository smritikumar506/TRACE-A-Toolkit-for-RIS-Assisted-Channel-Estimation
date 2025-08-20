import socket
import pickle
import numpy as np

# === Parameters ===
results_file = "a_phi_snr_results.txt"
UDP_IP = "127.0.0.1"
C_BPSKRX_UDP_PORT = 5021
CH_BPSKRX_UDP_PORT = 5030
BPSKRX_C_UDP_PORT = 5035
Message_Signal = b"Rx Message Data"
MSG_SNR = b"SNR Rotated Data"

# === Socket Setup ===
sockcbpskrx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockcbpskrx.bind((UDP_IP, C_BPSKRX_UDP_PORT))

sockchbpskrx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockchbpskrx.bind((UDP_IP, CH_BPSKRX_UDP_PORT))

sockbpskrxc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# === Noise Addition ===
def add_ktb_noise(signal, B_hz, T):
    k = 1.38e-23
    noise_power = k * B_hz * T
    sigma = np.sqrt(noise_power)
    noise = (1000 * sigma / (np.sqrt(2)) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)))
    return signal + noise, noise

# === Communication parameters ===
Fs = 1e6  # Sampling frequency
sps = 10  # Samples per symbol
Ts = sps / Fs
B_hz = 1 / (2 * Ts)  # Bandwidth

print("BPSK Receiver Ready...")

while True:
    data, addr = sockcbpskrx.recvfrom(1024)
    if data == Message_Signal:
        print("Signal notification received from cCntroller.")
        data1, addr1 = sockchbpskrx.recvfrom(65535)
        data_unpack = pickle.loads(data1)
        y = np.array(data_unpack[0])     # Received signal matrix
        
        # === Add thermal noise ===
        y_noisy, noise = add_ktb_noise(y, B_hz, 290)

        # --- SNR Calculation ---
        # signal_power = np.mean(np.abs(y_noisy) ** 2)
        # SNR_dB = signal_power
        noise_power = np.max(np.abs(noise) ** 2)
        y_total = (np.abs(y_noisy))**2
        signal_power = np.mean(y_total)
        SNR_dB = 10 * np.log10(signal_power / noise_power)
        # SNR_dB = np.mean(np.abs(y) ** 2)

        # Send to Controller
        packet = [MSG_SNR, SNR_dB]
        packet_serialized = pickle.dumps(packet)
        sockbpskrxc.sendto(packet_serialized, (UDP_IP, BPSKRX_C_UDP_PORT))
        print(f"Sent to Controller SNR={SNR_dB:.4f} dB")

        # --- Simple BPSK detection ---
        detected_bits_list = []
        for y_elem in y_noisy:
            detected_bits = (y_elem.real > 0).astype(int)
            detected_bits_list.append(detected_bits)
        detected_bits_flat = np.concatenate(detected_bits_list).astype(int)
        print("Detected bits:", "".join(map(str, detected_bits_flat)))
