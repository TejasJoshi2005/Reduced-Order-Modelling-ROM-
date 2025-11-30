# Physics-Informed Reduced Order Modeling (ROM) for Fluid Dynamics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

##  Project Overview
This project develops a **Reduced Order Model (ROM)** to accelerate high-fidelity Computational Fluid Dynamics (CFD) simulations. By applying machine learning techniques—specifically **Proper Orthogonal Decomposition (POD)** and **Dynamic Mode Decomposition (DMD)**. I extracted the dominant flow structures from a massive dataset (1.8 million points per snapshot) generated on a High-Performance Computing (HPC) cluster.

The resulting model captures **95% of the flow energy** while offering a **1000x speedup** in prediction time, demonstrating the potential for real-time Digital Twin applications in aerospace and automotive industries.

##  Key Features
* **HPC-Scale Processing:** Automated pipeline to process unstructured text data (100GB+) on remote supercomputing clusters using Linux/Bash.
* **Dimensionality Reduction:** Compressed 1.8M spatial degrees of freedom into <20 dominant modes using SVD-based algorithms.
* **Unstructured-to-Structured Interpolation:** Implemented cubic interpolation (`scipy.interpolate.griddata`) to visualize and analyze unstructured mesh data.
* **Physics Extraction:** Isolated coherent structures (vortex shedding) and stability eigenvalues using exact DMD.

## 📊 Results & Visualization

### 1. Flow Reconstruction (POD)
 Comparison between the original high-fidelity snapshot and the reconstructed low-order model using only 13 modes.
 
<img width="2250" height="750" alt="reconstruction_snapshot_175" src="https://github.com/user-attachments/assets/6a3adb46-f314-4136-b602-8009264266ff" />



### 2. Energy Distribution
The cumulative energy plot shows that the first 10-15 modes capture over 95% of the flow physics, validating the efficiency of the ROM.

<img width="400" height="300" alt="cumulative_energy" src="https://github.com/user-attachments/assets/d2c21105-ffa2-4346-97f1-957c4e774925" />


### 3. Dynamic Mode Spectrum
The DMD eigenvalues plotted on the complex plane, identifying the stable and periodic (vortex shedding) frequencies of the system.

<img width="400" height="300" alt="DMD_eigenvalues_complex_plane" src="https://github.com/user-attachments/assets/4c2807a4-a63a-4645-813f-8edf82ef4f85" />


## 🛠️ Tech Stack
* **Languages:** Python (NumPy, SciPy, Matplotlib), Bash/Shell Scripting
* **Infrastructure:** High-Performance Computing (HPC), Linux Environment
* **Algorithms:** SVD, POD, Exact DMD, Radial Basis Function Interpolation
