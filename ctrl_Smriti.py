import socket
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


def save_results_to_file(data, filename="perturb_snr_a.txt"):
    with open(filename, "w") as f:
        for a_val, perturb, snr in data:
            f.write(f"{a_val:.6f},{perturb:.6f},{snr:.4f}\n")

def plot_per_a():
    # Load the data
    data = pd.read_csv("perturb_snr_a.txt", header=None, names=['a', 'perturbation', 'SNR'])

    # Get global min and max SNR values
    snr_min = data['SNR'].min()
    snr_max = data['SNR'].max()

    # Convert all columns to float just to be sure
    data = data.astype(float)

    # Plot
    plt.figure(figsize=(8, 5))

    # Group by 'a' and plot each group
    for a_val, group in data.groupby('a'):
        group = group.sort_values('perturbation')
        plt.plot(group['perturbation'].values, group['SNR'].values, marker='o', markersize=7, linestyle='--', linewidth=0.6, label=f'a = {a_val:.4f}')

        plt.xlabel("Perturbation", fontsize=30)
        plt.tick_params(axis='both', labelsize=24, width=2)
        plt.ylabel("SNR (dB)", fontsize=30)
        plt.ylim(snr_min-1, snr_max+1)  # Set y-axis limits
        # plt.title("SNR vs Perturbation for Different a")
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig("perturbation_snr.png")

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W        

# Function to rotate RIS phase vector
def rotate_phi(phi_vec, delta):
    N = phi_vec.shape[0]
    rotated_phi = np.zeros_like(phi_vec, dtype=complex)
    
    for i in range(N):
        delta_deg = delta + np.random.uniform(-100, 100) if delta != 0 else 0
        rotated_phi[i] = np.exp(1j * (np.angle(phi_vec[i]) + delta_deg * np.pi / 180))
    
    return rotated_phi


UDP_IP = "127.0.0.1"
CHANNEL_UDP_PORT = 5004
PILOT_UDP_PORT = 5005
BPSK_UDP_PORT = 5013
C_RX_UDP_PORT = 5007
RX_C_UDP_PORT = 5001
C_BPSKRX_UDP_PORT=5021
BPSKRX_C_UDP_Port=5035


K =1 # no. of Transmitter Antenna
M =1 # no. of Receiver Antenna

Sock_bpsk_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
Sock_channel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
Sock_pilot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
Sockrx_c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
Sockrx_c.bind((UDP_IP, RX_C_UDP_PORT))
sockcbpskrx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockbpskrxc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockbpskrxc.bind((UDP_IP,BPSKRX_C_UDP_Port))

# Learning phase
RIS_ROWS=2
RIS_COLUMNS=3
pilot_power = 1
msg_power = 1
Config_Number=RIS_ROWS*RIS_COLUMNS

#Prepare the channel for H and G and RIS
MESSAGE_Channel_H1 = b"Set Config H1: "
MESSAGE_Channel_H2 = b"Set Config H2: "
MESSAGE_Channel_RIS = b"Set Config RIS: " # Not clear
MESSAGE_Channel_config = b"Set Config: "
MESSAGE_PILOT = b"Transmit Pilot"
MESSAGE_OPTIMAL_CONFIG =b"RIS Optimal Config:"
MESSAGE_Data = b"Transmit data"
MESSAGE_Rx_Pilot = b"Receive Pilot"
MESSAGE_SEND_RESULTS = b"Send Results"
Message_Signal = b"Rx Message Data"
MSG_SNR = b"SNR Rotated Data"

perturb_snr_data  = []
snr_list = []
g_diff_norm_list = []
g_diff_phase_list = []
a_list =[]

n=0
results_file = "perturb_snr_a.txt"

RIS_SIZE_LIST=[MESSAGE_Channel_RIS,RIS_ROWS,RIS_COLUMNS,K,M]
try:        
    while True:
        print("\n Controller starting new optimization cycle......") 
        time.sleep(2)
        try:
            serialized_list = pickle.dumps(RIS_SIZE_LIST)
            sent = Sock_channel.sendto(serialized_list, (UDP_IP, CHANNEL_UDP_PORT))
        finally:
            print("\n Closing socket.... Sent RIS Dimensions to Channel")

        try:
            # H1 = np.random.randn(Config_Number, K) + 1j * np.random.randn(Config_Number, K)
            # H1 = np.ones(Config_Number, K) + 1j * np.ones(Config_Number, K)
            # H1_row = np.exp(1j * 2 * np.pi * np.arange(1, Config_Number + 1) / Config_Number)
            H1_row = np.exp(1j * np.linspace(0, np.pi / 2, Config_Number))
            H1 = np.tile(H1_row[:, np.newaxis], (1, K))  # Shape: (Config_Number, K)
            H1_local = H1.copy() # store locally for comparision later
            print("\n H1 is",H1)
            list_h1=[MESSAGE_Channel_H1]
            # Flatten H1 and append each element
            for val in H1.flatten():
                list_h1.append(val)
            serialized_list = pickle.dumps(list_h1)
            sent = Sock_channel.sendto(serialized_list, (UDP_IP, CHANNEL_UDP_PORT))
        finally:
            print("\n Closing socket..... H1 Sent to Channel")

        try:
            n = n+1
            a = np.random.uniform(0.1, 0.5)   # Random decimal number less than 1
            # a = 10 ** np.random.uniform(-3, -1)  # log-uniform between 1e-3 and 1e-1
            # H2 = np.random.randn(M, Config_Number) + 1j * np.random.randn(M, Config_Number)
            # chl_factor = np.random.uniform(0.6, 0.975)
            # a = chl_factor**n
            # a = 1
            H2 = np.ones((M, Config_Number)) * a + 1j * np.zeros((M, Config_Number))
            H2_local = H2.copy() # store locally for comparision later
            print("\n H2 is",H2)
            list_h2=[MESSAGE_Channel_H2]
            # Flatten H2 and append each element
            for val in H2.flatten():
                list_h2.append(val)
            serialized_list = pickle.dumps(list_h2)
            sent = Sock_channel.sendto(serialized_list, (UDP_IP, CHANNEL_UDP_PORT))
            a_list.append(a)
        finally:
            print("\n Closing socket.... H2 Sent to Channel")


        Phi=DFT_matrix(Config_Number)
        print("\nDFT Martix is:",Phi)

        # Start the training phase
        for i in range(Config_Number):
            Sock_pilot.sendto(MESSAGE_Rx_Pilot, (UDP_IP, C_RX_UDP_PORT))
            #send config to channel
            list_phi=[MESSAGE_Channel_config]
            for j in range(len(Phi[i])):
                list_phi.append(Phi[i][j])
            serialized_list = pickle.dumps(list_phi)
            sent = Sock_channel.sendto(serialized_list, (UDP_IP, CHANNEL_UDP_PORT))
            print("\n DFT Matrix Sent to Channel")
            #Ask pilot generator to send the pilot signal to channel
            list_pilot=[MESSAGE_PILOT, pilot_power]
            serialized_list1 = pickle.dumps(list_pilot)
            Sock_pilot.sendto(serialized_list1, (UDP_IP, PILOT_UDP_PORT))
            print("\n [Controller to Pilot Generator] Send Pilot to Receiver:")

        #Collect data for channel estimation after the end of the training phase
        Sock_pilot.sendto(MESSAGE_SEND_RESULTS, (UDP_IP, C_RX_UDP_PORT))
        data1, addr1 = Sockrx_c.recvfrom(65535)  # buffer size is 1024 bytes
        result1 = pickle.loads(data1)
        noise_power = result1[0]
        y = result1[1:]
        symbol_power = np.abs(y)**2 
        #This is being correctly parsed for each pilot signal we get one y per RIS element
        # So each y is an array after pickle loads which is stacked later for g_est as y_tensor (NXMXN), 
        # from each col. y[1] to y[N] sequence is extracted to be combined with h1 and h2
        print("[Controller] Received pilot Result: ", y)
        y_power = np.mean(symbol_power) 
        SNR_dB = float(10 * np.log10(y_power/noise_power))
        #********************************************************************************************
        # COMPOSITE CHANNEL ESTIMATION AND COMPARISON
        # Step 1: Reshape H1 and H2 properly
        # H1: shape (N, K), H2: shape (M, N)
        # Composite channel: For each RIS element i: outer product of H2[:, i] and H1[i, :]
        # g_true will be a list of M x K matrices

        # Step 1: Reshape H1 and H2 properly
        # H1: shape (N, K), H2: shape (M, N)
        # Composite channel: For each RIS element i: outer product of H2[:, i] and H1[i, :]
        # g_true will be a list of M x K matrices

        N = Config_Number  # Number of RIS configurations

        # Step 1: Compute true composite channel tensor (N, M, K)
        g_true_tensor = np.zeros((N, M, K), dtype=complex)
        for i in range(N):
            h1_i = H1_local[i, :].reshape(K, 1)      # shape: (K, 1)
            h2_i = H2_local[:, i].reshape(M, 1)      # shape: (M, 1)
            g_true_tensor[i] = h2_i @ h1_i.T         # shape: (M, K)

        # Step 2: Stack received outputs into y_tensor (N, M, K)
        y_tensor = np.stack(y)  # shape: (N, M, K)
        g_true_norm = g_true_tensor/ np.linalg.norm(g_true_tensor)
        # Step 3: MMSE estimation for each Rx-Tx pair
        g_est_mmse = np.zeros((M, K, N), dtype=complex)
        perturb_snr_list = []

        for rx in range(M):
            for tx in range(K):
                y_vec = y_tensor[:, rx, tx]  # shape: (N,)
                g_est_mmse[rx, tx, :] = (np.sqrt(pilot_power) / (N * pilot_power + 1)) * y_vec.conj().T @ Phi.conj()

        # Reshape for direct comparison: g_true (N, M, K), g_est_mmse (M, K, N)
        g_est_tensor = np.transpose(g_est_mmse, (2, 0, 1))  # shape: (N, M, K)
        g_est_norm  = g_est_tensor  / np.linalg.norm(g_est_tensor)

        # Compute Frobenius norm of (g_true - g_est) across all configs
        # g_diff_norm = float((np.linalg.norm(g_true_tensor - g_est_tensor)))
        g_diff_norm = float(np.linalg.norm(g_true_norm - g_est_norm))
        # g_diff_phase = float(np.mean((np.angle(np.exp(1j * (np.angle(g_true_norm/a) - np.angle(g_est_norm/a))))))) # Wrap to [-π, π]
        g_diff_phase = float(np.max(np.angle(np.exp(1j * (np.angle(g_true_tensor) - np.angle(g_est_tensor))))))

        # Log data
        snr_list.append(SNR_dB)
        g_diff_norm_list.append(g_diff_norm)
        g_diff_phase_list.append(g_diff_phase)
        print(f"\n [Log] SNR = {SNR_dB:.2f} dB, ||g_true - g_est|| = {g_diff_norm:.4e}, Phase Alinment error = {g_diff_phase:.4e}")

        # Step 4: Display results
        print("\n--- Composite Channel Per Config (True Values) ---")
        for i in range(N):
            print(f"Config {i+1}:")
            print("True g matrix:\n", g_true_tensor[i])
            # print("Received y matrix:\n", y_tensor[i], "\n")

        print("\n--- MMSE Estimate per Rx-Tx across RIS ---")
        for rx in range(M):
            for tx in range(K):
                print(f"g_est for Rx {rx}, Tx {tx}:\n", g_est_mmse[rx, tx])

        # print("\n--- Best RIS Configuration per Rx-Tx Pair Based on Minimum |g_est|² ---")

        best_indices = np.zeros((M, K), dtype=int)
        min_powers = np.zeros((M, K))
        phi_best_matrix = np.zeros((M, K, N), dtype=complex)  # Store phi vectors

        for rx in range(M):
            for tx in range(K):
                # g_est_rx_tx = g_est_mmse[rx, tx, :]      # shape: (N,)
                # g_est_power = np.abs(g_est_rx_tx)**2     # shape: (N,)
                # g_est_rx_tx_complex = np.abs(g_est_mmse[rx, tx, :]) * np.exp(1j * (2*np.pi - np.angle(g_est_mmse[rx, tx, :])))   # shape: (N,)
                g_est_rx_tx_complex = 1 * np.exp(1j * (2*np.pi - np.angle(g_est_mmse[rx, tx, :])))   # shape: (N,)
                g_est_rx_tx = 2*np.pi - np.angle(g_est_mmse[rx, tx, :])
                # g_est_rx_tx = (-1)* np.angle(g_est_mmse[rx, tx, :])      # shape: (N,)
                print(f"\n Rx {rx}, Tx {tx} -> Best Config: {g_est_mmse[rx, tx, :]}" )
                # print(f"phi_best: {g_est_rx_tx}\n")
                # print(f"or: {g_est_mmse[rx, tx, :]}\n")
                perturb_list = []
                phi_shifted =np.array([], dtype=complex)
                phi_shifts = [0] + list(range(-10, -46, -10)) + list(range(10, 46, 10))
                for delta in phi_shifts:
                    # delta = delta + np.random.uniform(-7, 7) if delta != 0 else delta
                    # print("Applied delta:", delta)
                    phi_shifted = rotate_phi(g_est_rx_tx, delta)
                    perturb_list.append(delta)
                    # Structure: [MESSAGE_HEADER, RIS_config (as list), N, K, M, pilot_power]
                    optimal_config_message = [MESSAGE_OPTIMAL_CONFIG, list(phi_shifted), Config_Number, K, M, msg_power]
                    serialized_opt = pickle.dumps(optimal_config_message)
                    Sock_channel.sendto(serialized_opt, (UDP_IP, CHANNEL_UDP_PORT))                 
                    print("\n[Controller] Sent  RIS config to channel.", phi_shifted)
                    # === Tell BPSK Receiver to send the message signal ===
                    print("\nInstructing BPSK Receiver to Rx data...")
                    sockcbpskrx.sendto(Message_Signal, (UDP_IP, C_BPSKRX_UDP_PORT))
                    # === Instruct BPSK transmitter to send the message signal ===
                    print("\nInstructing BPSK transmitter to send data...")
                    # Send msg_power....
                    message_to_bpsk = [MESSAGE_Data, msg_power]
                    serialized_bpsk_msg = pickle.dumps(message_to_bpsk)                  
                    Sock_bpsk_tx.sendto(serialized_bpsk_msg, (UDP_IP, BPSK_UDP_PORT))
                    
                    # --- Receive perturbation SNR results ---
                    data2, addr2 = sockbpskrxc.recvfrom(65535)
                    msg_unpack = pickle.loads(data2)
                    if msg_unpack[0] == b"SNR Rotated Data":
                        snr_val = float(msg_unpack[1])
                        current_a = a_list[-1]
                        perturb_snr_list.append((current_a, perturb_list[-1], snr_val))
                        print(f"\n [Controller] Got perturb={perturb_list[-1]}, SNR={snr_val} dB")        
        perturb_snr_data.extend(perturb_snr_list)
        time.sleep(.1)
except KeyboardInterrupt:
    print("\n[Controller] Stopped by user.")
    # Save to file after loop exits
    with open("snr_vs_gdiff.txt", "w") as f:
        for snr, gdiff, gphase in zip(snr_list, g_diff_norm_list, g_diff_phase_list):
            f.write(f"{snr:.4f}, {gdiff:.6e}, {gphase:.6e}\n")

    print("[Controller] Logged SNR vs g_diff to 'snr_vs_gdiff.txt'")

    # print("Norm_Difference:", g_diff_norm_list)
        
    # Create the figure and twin y-axes
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left Y-axis: Frobenius norm error
    # ax1.set_xlabel("SNR (dB)")
    ax1.set_xlabel("a", fontsize=30)
    ax1.set_ylabel("||g_true - g_est|| (Frobenius norm)", color='blue', fontsize=30)
    # ax1.plot(snr_list, g_diff_norm_list, 'o', color='blue', label="Frobenius Norm")
    ax1.plot(a_list, g_diff_norm_list, 'o', color='blue', markersize=3, label="Frobenius Norm")
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=24, width=2)
    ax1.tick_params(axis='x', labelsize=24, width=2) 

    # Right Y-axis: Phase error in radians
    ax2 = ax1.twinx()
    ax2.set_ylabel("Phase Error (radians)", color='red', fontsize=30)
    # ax2.plot(snr_list, g_diff_phase_list, 'o', color='red', label="Phase Error")
    ax2.plot(a_list, g_diff_phase_list, 'o', color='red', markersize=3, label="Phase Error")
    ax2.tick_params(axis='y', labelcolor='red', labelsize=24, width=2)

    # Formatting
    # plt.title("SNR vs Channel Estimation Error Metrics")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig("snr_vs_gdiff_and_phase_plot.png")

    # Save perturbation vs SNR data
    save_results_to_file(perturb_snr_data)
    print("[Controller] Saved perturbation-SNR data to 'perturb_snr_a.txt'")

    # Plot perturbation vs SNR for all a
    plot_per_a()

    # plt.show()