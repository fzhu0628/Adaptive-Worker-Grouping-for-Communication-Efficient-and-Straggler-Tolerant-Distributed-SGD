# Adaptive-Worker-Grouping-for-Communication-Efficient-and-Straggler-Tolerant-Distributed-SGD
## Overview
- This is my first published paper titled *Adaptive Worker Grouping for Communication-Efficient and Straggler-Tolerant Distributed SGD*.
- This paper was published at **IEEE International Symposium on Information Theory** (ISIT) in 2022.
- About this paper:
  - In this paper, we designed a novel algorithm named ***G-CADA*** aiming to improve the time and communication efficiencies of a distributed learning system.
  - Based on the famous CADA algorithm which lazily aggregates the gradients from workers to improve communication efficiency, G-CADA partitions the workers into groups (scheduled adaptively at each iteration) that are assigned the same datasets. Hence, at the cost of additional storage at the workers, the server *only waits for the fastest worker* in each selected group, thus increasing the robustness of the system to stragglers.
  - Numerical illustrations are provided to demonstrate the significant gains in time and communication efficiencies.
## Key ideas of G-CADA
- A total of $M$ workers are divided into $G$ groups, each with the same number of workers.
- The total dataset is also divided into $G$ shards, and each worker in each group is assigned one shard of the dataset, i.e., workers in the _same group_ have access to the _same shard of dataset_.
- Each group maintains a parameter $\tau_g^k$, indicating the "age" (number of rounds the group has not talked to the server) of group $g$ at iteration $k$.
- At iteration $k$, the server selects groups according to the condition:

  ![image](https://github.com/user-attachments/assets/c431e41b-0594-4fce-ae44-6f7a25cac91d)
  
  If the condition is violated, meaning that the change in the information of this group is sufficiently large, then the group is selected to talk to the server to update its information.


- An illustration of the algorithm:

 <img src="https://github.com/user-attachments/assets/22e1de71-5f33-45ad-8edd-6b1cc44facfa" width="40%" />

 - Pseudo code of the algorithm:

  ![image](https://github.com/user-attachments/assets/4cc7b57f-fa98-486e-81e6-666b4f5d38b0)




- For more information, we redirect the reader to our full paper: https://ieeexplore.ieee.org/abstract/document/9834752 or https://arxiv.org/abs/2201.04301.
## Results
- Numerical examples: 


 <img src="https://github.com/user-attachments/assets/1b72a073-6155-404c-9f27-0ba9f18bd2f2" width="40%" />

 Apparently, G-CADA outperforms the benchmarks in time efficiency, communication cost and computation cost.

## Codes
- In the file linear_regression_ps.py, we consider the *linear regression* model with **MNIST** dataset and *quadratic error loss* function.
- We compared our G-CADA algorithm with state-of-the-art algorithms such as distributed SGD, CADA, and distributed Adam.
- Results show that our algorithm achieves superiority over the benchmarks in terms of both time and communication efficiencies.
