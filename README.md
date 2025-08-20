# TRACE-A-Toolkit-for-RIS-Assisted-Channel-Estimation
TRACE is a toolkit for channel estimation in RIS assisted wireless systems, where the ideal RIS response and wireless channel are modeled. Toolkit includes an end-to-end generic transceiver and 2 phase CE process. An initial training phase estimates the cascaded channel, followed by data transmission with fixed optimal RIS settings. 
TRACE is a modular,
socket-based simulator comprising independent Transmitter,
Smart Radio Environment, Controller, and Receiver modules.
It implements a standard training phase through DFT-based
MMSE cascaded channel estimation, this is followed by
transmission of data under optimal RIS configuration, and a
Differential Channel-Aware RIS Update phase to adapt the RIS
phase shifts under minor channel perturbations without full
retraining. The socket-based infrastructure ensures complete
decoupling between estimation logic, channel modeling and
modulation strategy, supporting plug-and-play experimentation
which is directly extendable to real-time SDR-based deploy-
ments too.
