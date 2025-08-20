import socket
import numpy as np
import pickle
import math

UDP_IP = "127.0.0.1"
CHANNEL_UDP_PORT=5004
DATA_CHANNEL_UDP_PORT=5010
BPSK_CHANNEL_UDP_PORT=5020
CH_RX_UDP_PORT=5015
CH_BPSKRX_UDP_PORT=5030

#The above should come from Smriti ma’am’s simulator
sockcc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockcc.bind((UDP_IP, CHANNEL_UDP_PORT))
sockdc= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockdc.bind((UDP_IP, DATA_CHANNEL_UDP_PORT))
sockbc= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockbc.bind((UDP_IP, BPSK_CHANNEL_UDP_PORT))
sockchrx=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockchbpskrx=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

MESSAGE_Channel_H1 = b"Set Config H1: "
MESSAGE_Channel_H2 = b"Set Config H2: "
MESSAGE_Channel_RIS = b"Set Config RIS: "
MESSAGE_Channel_config = b"Set Config: "
MESSAGE_PILOT = b"Transmit Pilot"
MESSAGE_OPTIMAL_CONFIG =b"RIS Optimal Config:"
Message_Transmission =b"Message Transmitted"

Config_number = 0
Tx_no = 0
Rx_no = 0
RIS_ROWS= 0
RIS_COLUMNS= 0
Phi1 = np.array([], dtype=complex)

#Set the initial parameters for Channel
while True:
    data, addr = sockcc.recvfrom(1024) # buffer size is 1024 bytes
    
    try:
        received_list = pickle.loads(data)
    except pickle.UnpicklingError:
        print("Error: Could not unpickle received data.")
    except Exception as e:
        print(f"An error occurred: {e}")

    if(received_list[0]==MESSAGE_Channel_RIS):
        RIS_ROWS=received_list[1]
        RIS_COLUMNS=received_list[2]
        Tx_no =received_list[3]
        Rx_no =received_list[4]
        Config_number = RIS_ROWS * RIS_COLUMNS
    print("--------------------------------------------------------------------")
    
    if received_list[0] == MESSAGE_OPTIMAL_CONFIG:
        optimal_phi = np.array(received_list[1], dtype=complex)    #  RIS phase values
        Config_number = received_list[2]
        Tx_no = received_list[3]
        Rx_no = received_list[4]
        pilot_power = received_list[5]    
        print("\n Received RIS configuration.:", optimal_phi)

        # === Receive BPSK Data Signal ===
        sockbc.settimeout(2.0)  # Safety timeout
        try:
            data_bpsk, addr_bpsk = sockbc.recvfrom(65535)
            received_bpsk = pickle.loads(data_bpsk)
            if received_bpsk[0] == Message_Transmission:
                
                bpsk_signal = np.array(received_bpsk[1], dtype=np.complex64)
                sps = int(received_bpsk[2])
                sampled = bpsk_signal[::sps]    # Symbol-spaced sampling
                
                # Step 1: Reshape the received message stream into (L x K)
                # total_symbols = len(sampled)
                # L1 = int(total_symbols // (Tx_no * sps))  # Length of each stream
                # L = L1 *sps
                # X_matrix = sampled[:L*Tx_no].reshape((L, Tx_no))  # shape: (L, K)
                total_symbols = int(len(sampled)-1) // Tx_no
                X_matrix = sampled[:total_symbols*Tx_no].reshape((total_symbols, Tx_no))  # shape: (L, K)
                L = total_symbols

                # Step 2: Channel matrices
                H1_mat = H1_received        # shape: (N, K)
                H2_mat = H2_received        # shape: (M, N)
                phi_vec = np.array(optimal_phi if 'optimal_phi' in locals() else Phi, dtype=complex)

                # Compute channel for this shifted phi
                RIS_matrix = np.diag(phi_vec)
                G = H2_mat @ RIS_matrix @ H1_mat   # Effective channel: shape (M, K)
                # Apply channel to each symbol
                y1_output = []
                
                for n in range(L):
                    x_n = X_matrix[n, :].reshape((-1, 1))  # shape: (K, 1)
                    y_n = G @ x_n                          # shape: (M, 1)
                    y1_output.append(y_n.flatten())        # shape: (M,)

                y1_output = np.array(y1_output)  # shape: (L, M)
                # print(y1_output)
                # Send this result to BPSK receiver
                message_data = [y1_output]
                serialized_message = pickle.dumps(message_data)
                sockchbpskrx.sendto(serialized_message, (UDP_IP, CH_BPSKRX_UDP_PORT))


        except socket.timeout:
            print("Timeout: No BPSK signal received.")
        except Exception as e:
            print("Error processing BPSK data:", e)

    if(received_list[0]==MESSAGE_Channel_H1):
        H1=received_list[1:]
        H1_received = np.array(H1).reshape((Config_number, Tx_no))
        print("\nreceived instruction to set H1 to ...", H1, "\n")

    elif(received_list[0]==MESSAGE_Channel_H2): 
        H2=received_list[1:]
        H2_received = np.array(H2).reshape((Rx_no, Config_number))
        print("\nreceived instruction to set H2 to ... ",H2, "\n")
    elif received_list[0] == MESSAGE_Channel_config:
        Phi = received_list[1:]
        print("\nTraining RIS Configuration is  ... ", Phi,"\n")
    

        # Add a timeout to prevent hanging
        sockdc.settimeout(2.0)
        try:
            data1, addr1 = sockdc.recvfrom(1024)
            received_list1 = pickle.loads(data1)
            if received_list1[0] == MESSAGE_PILOT:
                pilot = received_list1[1:]
                print ("\n Pilot Received:", pilot)
                # Flatten and convert to array
                pilot_flat = np.array(pilot).flatten()

                # Average the I-Q samples of 1 symbol
                pilot_mean = np.mean(pilot_flat)

                # Create a repeated array of shape (Config_number, 1)
                pilot_received = np.full((Config_number, 1), pilot_mean, dtype=np.complex64)

                y_mimo = np.zeros((Rx_no, Tx_no), dtype=complex)

                # Form H1: shape (N, K), H2: shape (M, N), Phi: shape (N,)
                H1_mat = H1_received  # (N, K)
                H2_mat = H2_received  # (M, N)
                phi_vec = np.array(Phi).flatten()  # shape (N,)
                N = len(phi_vec)

                # Compute effective RIS-weighted channel matrix:
                RIS_matrix = np.diag(phi_vec)
                G = H2_mat @ RIS_matrix @ H1_mat  # Shape (M, K)

                # Transmit BPSK signal (1 symbol per Tx)  pilot is used here
                x = pilot_received.flatten()[:Tx_no]  # size: (K,)
                #-----------------------------------------------------
                y_mimo = G @ x.reshape(-1, 1)         # shape: (M, 1)

                print("Channel: G matrix = \n", G)
                print("Transmit x = ", x)
                print("Received y = \n", y_mimo)

                # Send y to receiver
                list_y = pickle.dumps(y_mimo)
                sockchrx.sendto(list_y, (UDP_IP, CH_RX_UDP_PORT))


                # print("Received pilot signal:\n", pilot_received)
            else:
                print("Unexpected message on data channel:", received_list1[0])
        except socket.timeout:
            print("Timeout: No pilot received after RIS config.")
        except Exception as e:
            print(f"Error receiving pilot: {e}")


