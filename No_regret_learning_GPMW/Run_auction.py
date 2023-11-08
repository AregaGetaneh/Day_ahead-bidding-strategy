import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from aux_functions import Bidder,  random_bidder, GPMW_bidder
from tqdm import tqdm
import pickle
import re

# a class to store auction data

class auction_data:
    def __init__(self):
        self.bids = []
        self.allocations = []
        self.payments = []
        self.marginal_prices = []
        self.payoffs = []
        self.regrets = []
        self.Q = []
        self.SW = []

# simulates the selection process in the auction

def optimize_alloc(bids, Q, cap):
    C = np.array([param[0] for param in bids])
    C = np.diag(C)
    D = np.array([param[1] for param in bids])
    n = len(bids)
    A = np.ones(n).T
    G = - np.eye(n)
    h = np.zeros(n)
    I = np.eye(n)

    # non-negativity doesn't strictly hold (small negative allocations might occur)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, C) + D.T @ x),
                      [G @ x <= h, A @ x == Q, I @ x <= cap])
    prob.solve()
    allocs = x.value
    social_welfare = prob.value
    # To fix very small values
    for i in range(len(allocs)):
        if allocs[i] < 10 ** (-5):
            allocs[i] = 0

    # only for quadratic case
    sample_winner = np.argmin(allocs)
    marginal_price = bids[sample_winner][0] * min(allocs) + bids[sample_winner][1]
    payments = marginal_price * allocs

    return allocs, marginal_price, payments, social_welfare


# runs a repeated auction

def run_auction(T, bidders, Q, cap, regret_calc):
    for b in bidders:
        b.restart()
    game_data = auction_data()
    for t in range(T):
        bids = []
        for bidder in bidders:
            action, ind = bidder.choose_action()
            bidder.played_action = action
            bidder.history_action.append(ind)
            bids.append(action)

        x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)

        # calculates payoffs from payments
        payoff = []
        for i, bidder in enumerate(bidders):
            payoff_bidder = payments[i] - (0.5 * bidder.cost[0] * x[i] + bidder.cost[1]) * x[i]
            payoff.append(payoff_bidder)
            bidder.history_payoff.append(payoff_bidder)
        game_data.payoffs.append(payoff)

        # calculates real regret for all bidders/ Hedge also needs this part for its update
        if regret_calc:
            regrets = []
            for i, bidder in enumerate(bidders):
                if i == len(bidders) - 1:
                    payoffs_each_action = []
                    for j, action in enumerate(bidder.action_set):
                        tmp_bids = bids.copy()
                        tmp_bids[i] = action
                        x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(tmp_bids, Q, cap)
                        payoff_action = payments_tmp[i] - (0.5 * bidder.cost[0] * x_tmp[i] + bidder.cost[1]) * x_tmp[i]
                        payoffs_each_action.append(payoff_action)
                        bidder.cum_each_action[j] += payoff_action
                    bidder.history_payoff_profile.append(np.array(payoffs_each_action))
                    regrets.append(
                        (max(bidder.cum_each_action) - sum(bidder.history_payoff))/(t+1))

            # update weights
            for i, bidder in enumerate(bidders):
                if bidder.type == 'GPMW':
                    bidder.update_weights(x[i], marginal_price)
            game_data.regrets.append(regrets)

        # store data
        game_data.Q.append(Q)
        game_data.SW.append(social_welfare)
        game_data.bids.append(bids)
        game_data.allocations.append(x)
        game_data.payments.append(payments)
        game_data.marginal_prices.append(marginal_price)

    return game_data


def combine(res, res_list):
    game_data_profile = []
    for name in res_list:
        with open(f'{name}.pckl', 'rb') as file:
            T = pickle.load(file)
            types = pickle.load(file)
            game_data_profile.append(pickle.load(file))
    
    total_len = len(game_data_profile)    
    combined_profile = []
    for i in range(len(types)):
        tmp = []
        for r in range(total_len):
            tmp += game_data_profile[r][i]
        combined_profile.append(tmp)
    
    with open(f'{res}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(combined_profile, file)


# simulates #num_games different repeated auction #num_runs times for different bidder types and averages each result

def simulate(num_games, num_runs, T, file_name):
    Q = 1448.4
    N = 5
    K = 10
    c_limit = 0.08
    d_limit = 10
    c_list = [ 
    [0.07, 0.08, 0.09, 0.10, 0.12, 0.075, 0.085, 0.095, 0.15, 0.17], 

    [0.02, 0.05, 0.06, 0.15, 0.25, 0.025, 0.15, 0.90, 0.13, 0.31], 

    [0.03, 0.04, 0.06, 0.12, 0.14, 0.095, 0.08, 0.09, 0.21, 0.27],
    
    [0.008, 0.01, 0.075, 0.24, 0.31,0.08, 0.09, 0.05, 0.11, 0.14],
    
    [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]]
    
    d_list = [ 
    [9, 10, 11.5, 14, 13, 10, 12, 13, 15, 11], 

    [10, 15, 12, 17, 11, 12, 14, 13, 65, 14], 

    [12, 14, 13, 16, 15, 13, 12, 15, 14, 12],
    
    [12, 14, 17, 15, 17, 14, 11, 17, 15, 12],

    [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]]
    
    cap = [700, 700, 700, 700, 700]
    
    max_payoff = 36000

    types = []
#     types.append('GPMW 0.1')
#     types.append('GPMW 0.5')
#     types.append('GPMW 1')
    types.append('GPMW 0.7')

    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
            other_bidders.append(random_bidder(c_list[i], d_list[i], K, has_seed=True))
            
        for type_idx, bidder_type in enumerate(types):
            match = re.match('(GPMW)\W?(\d+\.?\d*)?', bidder_type)
            if match:
                beta = match.groups()[1]
                if beta:
                    bidders = other_bidders + [GPMW_bidder(c_list[-1], d_list[-1], K, max_payoff, T, float(beta))]
            if bidder_type == 'Random':
                bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
                
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True))
                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)


def simulate_all_same(num_games, num_runs, T, file_name):
    types = ['All GPMW']
    Q = 1448.4
    N = 5
    K = 10
    c_limit = 0.08
    d_limit = 10
    c_list = [ 
    [0.07, 0.08, 0.09, 0.10, 0.12, 0.075, 0.085, 0.095, 0.15, 0.17], 

    [0.02, 0.05, 0.06, 0.15, 0.25, 0.025, 0.15, 0.90, 0.13, 0.31], 

    [0.03, 0.04, 0.06, 0.12, 0.14, 0.095, 0.08, 0.09, 0.21, 0.27],
    
    [0.008, 0.01, 0.075, 0.24, 0.31,0.08, 0.09, 0.05, 0.11, 0.14],
    
    [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]]
    
    d_list = [ 
    [9, 10, 11.5, 14, 13, 10, 12, 13, 15, 11], 

    [10, 15, 12, 17, 11, 12, 14, 13, 65, 14], 

    [12, 14, 13, 16, 15, 13, 12, 15, 14, 12],
    
    [12, 14, 17, 15, 17, 14, 11, 17, 15, 12],

    [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]] 
    cap = [700, 700, 700, 700, 700]
    
#     max_payoff = calc_max_payoff(Q, c_list, d_list, N, T, K, cap)
#     print(max_payoff)
    max_payoff = 36000
    beta = 0.7

    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        bidders = []
        for i in range(N):
            bidders.append(GPMW_bidder(c_list[i], d_list[i], K, max_payoff, T, beta))
                
        for run in tqdm(range(num_runs)):
            game_data_profile[0].append(run_auction(T, bidders, Q, cap, regret_calc=True))
                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)

# runs a repeated auction

def sim_GP_BR(num_runs, T, file_name):
    types = ['GPMW vs BR']
    game_data_profile = [[]]
    Q = 1448.4
    N = 5
    K = 10
    c_GP = [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095] #last player
    d_GP = [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]
    
    # Actions of others obtained from diagonalization + their true cost
    BR_profile = [(0.095, 13.0), (0.06, 12.0), (0.06, 13.0), (0.01, 14.0)]
    other_costs = [(0.07, 9), (0.03, 10), (0.03, 12), (0.008, 12)]
    
    cap = [700, 700, 700, 700, 700]
    max_payoff = 36000
    beta = 1
    
    GP_bidder = GPMW_bidder(c_GP, d_GP, K, max_payoff, T, beta)
    
    for run in tqdm(range(num_runs)):
        GP_bidder.restart()
        game_data = auction_data()
        for t in range(T):
            action, ind = GP_bidder.choose_action()
            GP_bidder.played_action = action
            GP_bidder.history_action.append(ind)
            bids = BR_profile + [action]
            x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)
            
            payoff = []
            for i in range(N):
                if i == N-1:
                    payoff_GP = payments[-1] - (0.5 * GP_bidder.cost[0] * x[-1] + GP_bidder.cost[1]) * x[-1]
                    GP_bidder.history_payoff.append(payoff_GP)
                else:
                    payoff_bidder = payments[i] - (0.5 * other_costs[i][0] * x[i] + other_costs[i][1]) * x[i]
                    payoff.append(payoff_bidder)
            game_data.payoffs.append(payoff)


            # calculates regret for the GP player
            bidder = GP_bidder
            i = -1
            payoffs_each_action = []
            for j, action in enumerate(bidder.action_set):
                tmp_bids = bids.copy()
                tmp_bids[i] = action
                x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(tmp_bids, Q, cap)
                payoff_action = payments_tmp[i] - (0.5 * bidder.cost[0] * x_tmp[i] + bidder.cost[1]) * x_tmp[i]
                payoffs_each_action.append(payoff_action)
                bidder.cum_each_action[j] += payoff_action
            bidder.history_payoff_profile.append(np.array(payoffs_each_action))
            regret = (max(bidder.cum_each_action) - sum(bidder.history_payoff))/(t+1)
            bidder.update_weights(x[i], marginal_price)
            # 1d regret put in a list to be compatible with the rest of the code
            game_data.regrets.append([regret])

            # store data
            game_data.Q.append(Q)
            game_data.SW.append(social_welfare)
            game_data.bids.append(bids)
            game_data.allocations.append(x)
            game_data.payments.append(payments)
            game_data.marginal_prices.append(marginal_price)
        game_data_profile[0].append(game_data)

    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)

