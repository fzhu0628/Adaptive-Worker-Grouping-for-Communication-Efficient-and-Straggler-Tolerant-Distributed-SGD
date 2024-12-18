# Adaptive-Worker-Grouping-for-Communication-Efficient-and-Straggler-Tolerant-Distributed-SGD
## Description
- This repository contains my first published paper titled *Adaptive Worker Grouping for Communication-Efficient and Straggler-Tolerant Distributed SGD*.
- This paper was published at **IEEE International Symposium on Information Theory** (2022).
- About this paper:
  - In this paper, we designed a novel algorithm named ***G-CADA*** aiming to improve the time and communication efficiencies of a distributed learning system.
  - Based on the famous CADA algorithm which lazily aggregates the gradients from workers to improve communication efficiency, G-CADA partitions the workers into groups (scheduled adaptively at each iteration) that are assigned the same datasets. Hence, at the cost of additional storage at the workers, the server *only waits for the fastest worker* in each selected group, thus increasing the robustness of the system to stragglers.
  - Numerical illustrations are provided to demonstrate the significant gains in time and communication efficiencies.
- An illustration of the algorithm: [Algorithm illustration](https://github.com/user-attachments/files/18185707/fig3.pdf).

- For more information, we redirect the reader to our full paper: https://ieeexplore.ieee.org/abstract/document/9834752.
## Codes
- In the file linear_regression_ps.py, we consider the *linear regression* model with **MNIST** dataset and *quadratic error loss* function.
- We compared our G-CADA algorithm with state-of-the-art algorithms such as distributed SGD, CADA, and distributed Adam.
- Results show that our algorithm achieves superiority over the benchmarks in terms of both time and communication efficiencies.
## Results
- Numerical examples: [Results](https://github.com/user-attachments/files/18185712/figf.pdf).
