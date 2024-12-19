# Adaptive-Worker-Grouping-for-Communication-Efficient-and-Straggler-Tolerant-Distributed-SGD

## Overview

- This repository showcases my first published paper titled *Adaptive Worker Grouping for Communication-Efficient and Straggler-Tolerant Distributed SGD*.
- The paper was presented at the **IEEE International Symposium on Information Theory (ISIT)** in 2022.

### About the Paper

- We propose a novel algorithm, ***G-CADA***, designed to enhance both time and communication efficiencies in distributed learning systems.
- Building upon the widely recognized CADA algorithm, which aggregates gradients from workers lazily to improve communication efficiency, G-CADA introduces adaptive worker grouping. Specifically:
  - Workers are partitioned into groups dynamically at each iteration, and each group is assigned the same dataset shard.
  - By utilizing additional storage at the workers, the server interacts with only the fastest worker in each group, significantly improving robustness against stragglers.
- Numerical simulations demonstrate substantial gains in both time and communication efficiency.

## Key Concepts of G-CADA

- **Group formation**:

  - A total of \$M\$ workers are divided into \$G\$ groups, with each group containing an equal number of workers.
  - The dataset is split into \$G\$ shards, and all workers within a group access the same shard.

- **Age parameter**:

  - The server maintains a parameter, \$τ\_g^k\$ for each group, representing the “age” (i.e., the number of rounds the group has not communicated with the server) at iteration \$k\$.

- **Group selection**:

  - At each iteration \$k\$, groups are selected based on a predefined condition shown below. If the condition is violated, indicating that the group’s information has changed significantly, the group communicates with the server to update its parameters.

     ![image](https://github.com/user-attachments/assets/c431e41b-0594-4fce-ae44-6f7a25cac91d)
  
- **Age update**:

  - Groups that communicate with the server reset their age parameter.

### Algorithm Illustration

An example visualization of the algorithm:

<img src="https://github.com/user-attachments/assets/22e1de71-5f33-45ad-8edd-6b1cc44facfa" width="40%" />

### Pseudocode

The pseudocode below provides a step-by-step representation of the G-CADA algorithm:

![image](https://github.com/user-attachments/assets/4cc7b57f-fa98-486e-81e6-666b4f5d38b0)


- For more detailed information, please refer to our full paper: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9834752) or [arXiv](https://arxiv.org/abs/2201.04301).

## Results

- **Numerical examples**:

<img src="https://github.com/user-attachments/assets/1b72a073-6155-404c-9f27-0ba9f18bd2f2" width="40%" />

* G-CADA demonstrates clear advantages over benchmark algorithms in terms of time efficiency, communication cost, and computational cost.

## Code Description

- The implementation in `linear_regression_ps.py` simulates a *linear regression* model using the **MNIST** dataset and a *quadratic error loss* function.
- **Benchmark comparisons**:
  - G-CADA is compared against state-of-the-art algorithms, including distributed SGD, CADA, and distributed Adam.
  - Results consistently show that G-CADA outperforms these methods in both time and communication efficiencies.

