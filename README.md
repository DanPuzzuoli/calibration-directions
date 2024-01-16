# Paper supplement: Simple calibration methods for numerically designed quantum gates

This repository is a code supplement to the paper Simple calibration methods for numerically
designed quantum gates **\<add hyperlink once submitted\>**. The notebook used to optimize the pulse
used in the paper that is robust to frequency and amplitude variations is given in the
``freq_amp_robust_pulse_optimization`` folder, and the code for the spectator robust pulse is in the
``spectator_pulse_optimization`` folder. Note that the latter code was set up to run using the
IBM-internal Cognitive Compute Cluster high performance computing resource.

The code in this repository can be run with the following environment:
- Python 3.11
- Qiskit Dynamics 0.4.3
- Qiskit 0.45.1
- JAX 0.4.6
