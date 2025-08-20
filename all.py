import subprocess
import time


print(f"Launching BPSKRx...")
subprocess.Popen(['gnome-terminal', '--tab', '--', 'bash', '-c', 'python3 BPSKRx.py; exec bash'])
time.sleep(0.5)
print(f"Launching BPSKTx...")
subprocess.Popen(['gnome-terminal', '--tab', '--', 'bash', '-c', 'python3 BPSKTx.py; exec bash'])
time.sleep(0.5)
print(f"Launching rx_Smriti...")
subprocess.Popen(['gnome-terminal', '--tab', '--', 'bash', '-c', 'python3 rx_Smriti.py; exec bash'])
time.sleep(0.5)
print(f"Launching pil_Smriti...")
subprocess.Popen(['gnome-terminal', '--tab', '--', 'bash', '-c', 'python3 pil_Smriti.py; exec bash'])
time.sleep(0.5)
print(f"Launching chl_Smriti...")
subprocess.Popen(['gnome-terminal', '--tab', '--', 'bash', '-c', 'python3 chl_Smriti.py; exec bash'])
time.sleep(0.5)
print(f"Launching ctrl_Smriti...")
subprocess.Popen(['gnome-terminal', '--tab', '--', 'bash', '-c', 'python3 ctrl_Smriti.py; exec bash'])
