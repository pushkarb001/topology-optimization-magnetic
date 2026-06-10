Topology Optimization of a 2D Magnetic Actuator using Evolutionary Computation

This project implements topology optimization on a C-core electromagnet using two shape parameterization methods — Normalized Gaussian Networks (NGN) and Radial Basis Functions (RBF) — with a Genetic Algorithm as the optimizer and FEMM as the electromagnetic simulator.
Built as preparation for a thesis project on coaxial magnetic gear optimization. The core idea is the same as the professor's work — use NGN/RBF to represent the iron shape as a continuous density field, threshold it to a binary iron/air map, simulate it with FEMM, and let the GA evolve the shape toward maximum flux linkage.


What this project does

The design region is the right side of a C-core electromagnet. Instead of fixing the iron shape (two tips with a gap), the optimizer is free to place iron anywhere in that region. The GA searches over weight vectors that define the density field, and FEMM evaluates each candidate shape by computing the magnetic flux linkage through the coil.
Best result: NGN matched the solid iron baseline at 0.008733 Wb, while RBF reached only 0.002752 Wb. NGN is 3.2x better on this problem.
