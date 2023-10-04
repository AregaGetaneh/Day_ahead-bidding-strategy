# Learning to Bid in Forward Electricity Markets with Minimal Information

This repository contains the code and input data associated with the paper titled 'Learning to Bid in Forward Electricity Markets with Minimal Information' authored by Arega Getaneh Abate, Dorsa Majdi, Jalal Kazempour, and Maryam Kamgampour.

### Guide to the code

This README file is divided into the following sections:
•	General comments
•	Auxillary functions
•	Run Auction functions
•	Diagonalization method
•	Plottings

### Important general comments

The code begins by importing several Python libraries, including NumPy, cvxpy, scikit-learn (for Gaussian process regression), Matplotlib, tqdm (for progress bars), pickle (for data serialization), re (for regular expressions), gurobipy, linopy, and pyomo.environ (for the optimization or diagonalization method).
The code is divided into two main parts: the learning part and the diagonalization for the benchmark mode. The learning algorithm's code is split across three files: auc_function, Run_auction, and Plottings. On the other hand, the Diagonalization method is implemented within a single file, which includes the solution of the EPEC problem using the diagonalization algorithm and the market operator's computation considering trustful bids from each bidder.

### Auxillary functions

The auc_function file contains the following classes:
•	A parent Bidder class is defined where we defined the common definitions for both Random and GPMW bidders.
•	random_bidder class: Represents a bidder with random bidding behavior.
•	GPMW_bidder class: Implements a bidding strategy using Gaussian Process Multiplicative Weight update algorithm. Importantly, this class contains some functions important for GP such as how it updates during the auction rounds using gpr.fit and gpr.predict for Gaussian process regression and Gaussian process prediction.

### Run Auction functions

The 'Run_auction' file contains crucial functions for the auction environment, including the embedded optimization function. Within this file, the 'run_auction' function calculates the payoff and regret of the learning bidder. It also stores essential data such as bids (c and d values), allocations, marginal prices, and social welfare (the Market Operator's objective function) for post analysis.
Additionally, the file includes three essential functions: 'simulate' for the GP-Random case, where Bidder 5 uses GPMW and Bidders 1-4 use the Random algorithm, 'simulate_all_same' for the case where all bidders use GPMW, and finally, 'sim_GP_BR' for the case where Bidder 5 uses GPMW while Bidders 1-4 employ their best responses obtained from the diagonalization algorithm.

### Diagonalization method

This file contains the code for the benchmark model. First, it defines the Mixed-Integer Quadratic Programming (MIQP) for N bidders. Then, it contains a function that defines the Diagonalization (Algorithm 2) algorithm as described in the paper. Subsequently, by considering the optimal bids of each bidder, the 'optimal_alloc' function solves the Market Operator's problem, determining allocations for every bidder and the market clearing price. It also addresses the optimization problem with the trustful bidding of each bidder, achieved by replacing the diagonalization results with the true cost functions of each bidder in the 'bids' tuple data.

### Plottings

The 'Plotting' file contains several plotting functions. These functions are designed to plot regret (for Bidder 5), social cost, and market clearing prices (for the system), as well as bid prices and payoffs (for Bidders 1 and 5). All of these plotting functions take into account three learning cases: all bidders using GPMW, Bidder 5 using GPMW while Bidders 1-4 use Random, and Bidder 5 using GPMW while Bidders 1-4 use their best responses from the diagonalization algorithm. Furthermore, the functions contain two optimization cases: diagonalization and trustful biddings.
The code runs only for Bidder 1 in the case where we need to plot for Bidder 5 as well. This is the case for bid prices and payoffs functions.  Hence, it is possible to plot for Bidder 5 by uncommenting the necessary lines in each function where there are comments indicating this.

### Summary

All the necessary libraries should be installed and imported properly for the smooth running of the code. The simulation results should be stored properly; indeed, we also include them in this repository for convenience. Finally, uncommenting and commenting the necessary lines of code and data is crucial. 

