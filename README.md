# Algorithms for the Subspace-Consensus Dominating Skyline query
This repository contains the C++ code of the implementation of two algorithms (SCDS and BL) for processing the Subspace-Consensus Dominating Skyline query, proposed in the related paper. This query type combines bounded output (k items), no mandatory scoring function, robustness across a family of user-selected attribute subspaces, and domination reasoning.

The SCDS algorithm is an efficient algorithm that employs index structures, sorting-based skyline methods, pruning rules, bounds, and a top-k heap. 

The BL algorithm is a baseline algorithm, designed under a brute-force approach, which can be used as a correctness-validation mechanism.

There are four RUN modes which can be used in the main function of the code:

1 = run SCDS on synthetic data

2 = run SCDS on data loaded from a .csv file

3 = run BL on synthetic data

4 = run BL on data loaded from a .csv file

(the desired parameters can be defined inside the main function).
