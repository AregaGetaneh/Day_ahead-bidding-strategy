import numpy as np
import numpy as np
import cvxpy as cp
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from linopy import Model
import pyomo.environ as pyo
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import random


def solve_miqp_problem(opt_unit,Q_t, 
                       T, 
                       N,
                       c_i, d_i, 
                       c_initial,
                       d_initial,
                       c_options,
                       d_options, 
                       capacity_df):


    model = pyo.ConcreteModel()

    model.N = pyo.RangeSet(0, N - 1)
    model.T = pyo.RangeSet(0, T - 1)
    model.C = pyo.RangeSet(0, len(c_options[0]) - 1)

    M_d = 500  
    M_p = 1000 

    # Primary, dual and binary variables
    model.x = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, name="x")
    model.mu_bar = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, name="mu_bar")
    model.mu_underline = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, name="mu_underline")
    model.lambda_t = pyo.Var(model.T, domain=pyo.Reals, name="lambda")
    model.psi_bar = pyo.Var(model.N, model.T, domain=pyo.Binary, name="psi_bar")
    model.psi_underline = pyo.Var(model.N, model.T, domain=pyo.Binary, name="psi_underline")

    # Binary variables for u and auxiliary variables
    model.u = pyo.Var(model.C, domain=pyo.Binary, name="u")
    model.c_prime = pyo.Var(domain=pyo.NonNegativeReals, name="c_prime")
    model.d_prime = pyo.Var(domain=pyo.NonNegativeReals, name="d_prime")

    # constraints
    model.c_prime_constr = pyo.Constraint(expr=model.c_prime == pyo.quicksum(c_options[opt_unit][i] * model.u[i] for i in model.C))
    model.d_prime_constr = pyo.Constraint(expr=model.d_prime == pyo.quicksum(d_options[opt_unit][i] * model.u[i] for i in model.C)) # Why C?
    model.u_constr = pyo.Constraint(expr=pyo.quicksum(model.u[i] for i in model.C) == 1)

    
    def cap_constr_rule(model, i, t):
        return capacity_df.iat[i, t] - model.x[i, t] >= 0

    model.cap_constr = pyo.Constraint(model.N, model.T, rule=cap_constr_rule)

    def psi_underline_constr_rule(model, i, t):
        return model.x[i, t] <= (1 - model.psi_underline[i, t]) * M_p

    model.psi_underline_constr = pyo.Constraint(model.N, model.T, rule=psi_underline_constr_rule)

    def psi_bar_constr_rule(model, i, t):
        return capacity_df.iat[i, t] - model.x[i, t] <= (1 - model.psi_bar[i, t]) * M_p
        #Cap is not time dependent
    model.psi_bar_constr = pyo.Constraint(model.N, model.T, rule=psi_bar_constr_rule)

    def mu_bar_constr_rule(model, i, t):
        return model.mu_bar[i, t] <= model.psi_bar[i, t] * M_d

    model.mu_bar_constr = pyo.Constraint(model.N, model.T, rule=mu_bar_constr_rule)

    def mu_underline_constr_rule(model, i, t):
        return model.mu_underline[i, t] <= model.psi_underline[i, t] * M_d

    model.mu_underline_constr = pyo.Constraint(model.N, model.T, rule=mu_underline_constr_rule)  

    def lambda_constr_rule(model, i, t):
            
    
        if i == opt_unit:
            return (model.c_prime * model.x[opt_unit, t] + model.d_prime - model.lambda_t[t]
                - model.mu_underline[opt_unit, t] + model.mu_bar[opt_unit, t] == 0)
        else:
            return (c_initial[i] * model.x[i, t] + d_initial[i] - model.lambda_t[t] - model.mu_underline[i, t] + model.mu_bar[i, t] == 0)

    model.lambda_constr = pyo.Constraint(model.N, model.T, rule=lambda_constr_rule)
    

    def q_constr_rule(model, t):
        return Q_t[t] - sum(model.x[i, t] for i in model.N) == 0

    model.q_constr = pyo.Constraint(model.T, rule=q_constr_rule)

    # Objective function
    def objective_rule(model):
        expr = 0
        expr += pyo.quicksum(model.lambda_t[t] * Q_t[t] for t in model.T)
        for i in model.N:
            if i == opt_unit:
                expr -= pyo.quicksum(c_i[i] * model.x[i, t] ** 2 + d_i[i] * model.x[i, t] for t in model.T)
            else:
                expr -= pyo.quicksum(
                    model.mu_bar[i, t] * capacity_df.iat[i, t]
                    + c_initial[i] * model.x[i, t] ** 2 + d_initial[i] * model.x[i, t] for t in model.T)
        return expr


    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Solve the model
    solver = pyo.SolverFactory("gurobi")
    results = solver.solve(model, options={"NonConvex": 2})

    # check if model solved to optimality
    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("Model is infeasible")
        return
    else:
        # check for complementary slackness
        for i in model.N:
            for t in model.T:
                temp = model.mu_bar[i, t].value * (capacity_df.iat[i, t] - model.x[i, t].value)
                #print(f"First condition: {temp}")
                if temp !=0:
                    print("Complementary slackness not satisfied for first")
                temp = model.mu_underline[i, t].value * model.x[i, t].value
                #print(f"Second condition: {temp}")
                if temp !=0:
                    print("Complementary slackness not satisfied for second")

    # return model.x values as a numpy array
    x_values = np.zeros((N, T))
    for i in model.N:
        for t in model.T:
            x_values[i, t] = model.x[i, t].value

    lambda_values = np.zeros(T)
    for t in model.T:
        lambda_values[t] = model.lambda_t[t].value
    
#     model.pprint()

    return model.c_prime.value, model.d_prime.value, x_values, lambda_values


def diagonalization_algorithm(N, T, Q_t, c_options, d_options, c_initial, d_initial,
    capacity_df, epsilon=1e-4, max_iterations=100):
    
    c_previous = c_initial[:]
    d_previous = d_initial[:]

    c_final = c_initial[:]
    d_final = d_initial[:]

    iterations = 0
    while True:
        for i in range(N):
            opt_agent = i
            c_prime, d_prime, x_values, lambda_values = solve_miqp_problem(
                opt_unit=opt_agent,
                Q_t = Q_t,
                T =T,
                N = N,
                c_i=c_i,
                d_i=d_i,
                c_initial=c_final,
                d_initial=d_final,
                c_options=c_options,
                d_options=d_options,
                capacity_df=capacity_df,
            )
            c_final[opt_agent] = c_prime
            d_final[opt_agent] = d_prime

        print(f"Current iteration: {iterations+1}")
        print(f"Previous c values: {c_previous}")
        print(f"Previous d values: {d_previous}")
        print()
        print(f"Updated c values: {c_final}")
        print(f"Updated d values: {d_final}")

        if iterations > max_iterations:
            break

        if (
            abs(np.array(c_final) - np.array(c_previous)).sum() < epsilon
            and abs(np.array(d_final) - np.array(d_previous)).sum() < epsilon
        ):
            print(f"Convergence achieved in {iterations+1} iterations.")
            break

        c_previous = c_final
        d_previous = d_final

        iterations += 1

    return c_final, d_final, x_values, lambda_values

# Input parameters
T = 1  # Number of time steps
N = 5  # Number of players (excluding the leader)


c_initial = [0.07, 0.02, 0.03, 0.008, 0.01, 0.07, 0.02, 0.03, 0.008, 0.01]  # Othe generator cost
d_initial = [9, 10, 12, 12, 11, 9, 10, 12, 12, 11]  # other generators cost

c_options = [
    [0.07, 0.08, 0.09, 0.10, 0.12, 0.075, 0.085, 0.095, 0.15, 0.17], 

    [0.02, 0.05, 0.06, 0.15, 0.25, 0.025, 0.15, 0.90, 0.13, 0.31], 

    [0.03, 0.04, 0.06, 0.12, 0.14, 0.095, 0.08, 0.09, 0.21, 0.27],
    
    [0.008, 0.01, 0.075, 0.24, 0.31,0.08, 0.09, 0.05, 0.11, 0.14],
    
    [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]]

d_options = [
    [9, 10, 11.5, 14, 13, 10, 12, 13, 15, 11], 

    [10, 15, 12, 17, 11, 12, 14, 13, 65, 14], 

    [12, 14, 13, 16, 15, 13, 12, 15, 14, 12],
    
    [12, 14, 17, 15, 17, 14, 11, 17, 15, 12],

    [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]]
 

c_i = [0.07, 0.02, 0.03, 0.008, 0.01]  # all generators trustful bidding including i
d_i = [9, 10, 12, 12, 11]
Q_t = [1448.4]



capacity = [700, 700, 700, 700, 700]
capacity_df = pd.DataFrame(data=0, columns=list(range(N)), index=list(range(T)))

for i in range(N):
    capacity_df.iloc[:, i] = capacity[i]
capacity_df = capacity_df.T

c_final, d_final, x_values, lambda_values = diagonalization_algorithm(
    N = N,
    T = T,
    Q_t=Q_t,
    c_options=c_options,
    d_options=d_options,
    c_initial=c_initial,
    d_initial=d_initial,
    capacity_df=capacity_df,
    max_iterations=100)

print("Generation:")
for n in range(N):
    print(f"Agent {n+1}: {x_values[n]}")

print()
print("Lambda values:")
print(lambda_values)

# Market operators problem

def optimize_alloc(bids, Q, cap):
    # Extract bid parameters
    C = np.array([param[0] for param in bids])
    C = np.diag(C)
    D = np.array([param[1] for param in bids])
    n = len(bids)
    
    # Define optimization variables and constraints
    A = np.ones(n).T
    G = -np.eye(n)
    h = np.zeros(n)
    I = np.eye(n)

    # Define the optimization variable x (allocations)
    x = cp.Variable(n)

    # Define the optimization problem
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, C) + D.T @ x),
                      [G @ x <= h, A @ x == Q, I @ x <= cap])
    prob.solve()
    
    # Extract the optimized allocations
    allocs = x.value
    
    # To fix very small values
    for i in range(len(allocs)):
        if allocs[i] < 10 ** (-5):
            allocs[i] = 0

    # Calculate the marginal price and payments
    sample_winner = np.argmin(allocs)
    marginal_price = bids[sample_winner][0] * min(allocs) + bids[sample_winner][1]
    payments = marginal_price * allocs

    objective_result = 0.0
    for i in range(len(allocs)):
        objective_result += 0.5 * bids[i][0] * allocs[i] ** 2 + bids[i][1] * allocs[i]

    return allocs, marginal_price, payments, objective_result


    
bids = [(0.095, 13.0), (0.06, 12.0), (0.06, 13.0), (0.01, 14.0), (0.02, 12.0)]
Q = 1448.4
cap = [700.0, 700.0, 700.0, 700, 700]

# Call the function to solve the problem
allocs, marginal_price, payments, objective_result = optimize_alloc(bids, Q, cap)

# Print the results
print("Allocations:", allocs)
print("Marginal Price:", marginal_price)
print("Payments:", payments)
print("Objective Function Result:", objective_result)


# Truthfull bidding

def optimize_alloc(bids, Q, cap):
    # Extract bid parameters
    C = np.array([param[0] for param in bids])
    C = np.diag(C)
    D = np.array([param[1] for param in bids])
    n = len(bids)
    
    # Define optimization variables and constraints
    A = np.ones(n).T
    G = -np.eye(n)
    h = np.zeros(n)
    I = np.eye(n)

    # Define the optimization variable x (allocations)
    x = cp.Variable(n)

    # Define the optimization problem
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, C) + D.T @ x),
                      [G @ x <= h, A @ x == Q, I @ x <= cap])
    prob.solve()
    allocs = x.value
    
    # To fix very small values
    for i in range(len(allocs)):
        if allocs[i] < 10 ** (-5):
            allocs[i] = 0

    # Calculate the marginal price and payments
    sample_winner = np.argmin(allocs)
    marginal_price = bids[sample_winner][0] * min(allocs) + bids[sample_winner][1]
    payments = marginal_price * allocs

    objective_result = 0.0
    for i in range(len(allocs)):
        objective_result += 0.5 * bids[i][0] * allocs[i] ** 2 + bids[i][1] * allocs[i]

    return allocs, marginal_price, payments, objective_result


    
bids = [(0.07, 9.0), (0.02, 10.0), (0.03, 12.0), (0.008, 12.0), (0.01, 11.0)]
Q = 1448.4
cap = [700.0, 700.0, 700.0, 700, 700]

allocs, marginal_price, payments, objective_result = optimize_alloc(bids, Q, cap)


