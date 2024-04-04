import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from aux_functions import Bidder,  random_bidder, GPMW_bidder
from tqdm import tqdm
import pickle
import re


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
        self.final_dist = None
        
        
        
        # estimates maximum payoff from results of a random play

def calc_max_payoff(Q, c_list, d_list, N, T, K, cap):
    num_games = 10
    num_runs = 10
    game_data_profile = []
    for i in range(num_games):
        bidders = []
        for i in range(N):
            bidders.append(random_bidder(c_list[i], d_list[i], K))
        for run in range(num_runs):
            game_data_profile.append(run_auction(T, bidders, Q, cap, regret_calc=False).payoffs)
    return np.max(np.array(game_data_profile))


# simulates the selection process in the auction

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

def run_auction(T, bidders, Q, cap, regret_calc, regret_all=False):
    for b in bidders:
        b.restart()
    
    game_data = auction_data()
#     player_final_dists = []
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
                if regret_all or (not regret_all and i == len(bidders) - 1):
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
#                     bidder.update_weights(bidder.history_payoff_profile[-1])

            # update weights
            for i, bidder in enumerate(bidders):
                if bidder.type == 'Hedge':
                    bidder.update_weights(bidder.history_payoff_profile[t])
                if bidder.type == 'EXP3':
                    bidder.update_weights(bidder.history_action[t], bidder.history_payoff[t])
                if bidder.type == 'GPMW':
                    bidder.update_weights(x[i], marginal_price)
            game_data.regrets.append(regrets)

        # store data
        game_data.Q.append(Q)
        game_data.SW.append(social_welfare)
        game_data.bids.append(bids)
        game_data.allocations.append(x)
        game_data.payments.append(payments)
        game_data.payoffs.append(payoff)
        game_data.marginal_prices.append(marginal_price)
    game_data.final_dist = bidders[len(bidders) - 1].weights
    
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
        
        
    # Case a ==> Trustful Vs Hedge (Others play Trustful vs B5 Plays Hedge)  


# Case a ==> Trustful Vs Hedge 

def Trustful_vs_Hedge(num_games, num_runs, T, file_name):
    types = ['Trustful vs Hedge']
    game_data_profile = [[]]
    Q = 1448.4
    N = 5
    K = 10
    c_cost_Hedge = [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095] #last player
    d_cost_Hedge = [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]
 
    
    # Actions of others obtained from diagonalization + their true cost
    HG_profile = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]
    other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]
    

    
    cap = [700, 700, 700, 700, 700]
    max_payoff = 36000
    
    
    hedge_bidder = Hedge_bidder(c_cost_Hedge, d_cost_Hedge, K, max_payoff, T)
    
    player_final_dists = []
    for run in tqdm(range(num_runs)):
        hedge_bidder.restart()
        game_data = auction_data()
        for t in range(T):
            action, ind = hedge_bidder.choose_action()
#             action, ind = (0.02, 12), 5
            hedge_bidder.played_action = action
            hedge_bidder.history_action.append(ind)
            bids = HG_profile + [action]
            x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)
            
            payoff = []
            for i in range(N):
                if i == N-1:
                    payoff_HG = payments[-1] - (0.5 * hedge_bidder.cost[0] * x[-1] + hedge_bidder.cost[1]) * x[-1]
                    hedge_bidder.history_payoff.append(payoff_HG)
                    payoff.append(payoff_HG)
#                     if t==T-1:
#                         print(f'Run {run}) Hedge player payoff at round {t+1}: {payoff_HG}')
                else:
                    payoff_bidder = payments[i] - (0.5 * other_costs[i][0] * x[i] + other_costs[i][1]) * x[i]
                    payoff.append(payoff_bidder)
            game_data.payoffs.append(payoff)



            bidder = hedge_bidder
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
            bidder.update_weights(bidder.history_payoff_profile[-1])

            game_data.regrets.append([regret])

            # store data
            game_data.Q.append(Q)
            game_data.SW.append(social_welfare)
            game_data.bids.append(bids)
            game_data.allocations.append(x)
            game_data.payments.append(payments)
            game_data.marginal_prices.append(marginal_price)
        player_final_dists.append(hedge_bidder.weights)
        game_data_profile[0].append(game_data)

    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
        pickle.dump(player_final_dists, file)
        
    ## Case b ==> Trustful Vs Random (others play Trustful vs B5 Plays Random)

# Case b ==> Trustful Vs Random
def Trustful_vs_Random(num_games , num_runs, T, file_name):
    types = ['Trustful vs Random']
    game_data_profile = [[]]
    Q = 1448.4
    N = 5
    K = 10
    c_cost_Random = [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095] 
    d_cost_Random = [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]
 

    
    Trustful_profile = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]
    other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]
    

    

    
    cap = [700, 700, 700, 700, 700]
    max_payoff = 36000
    
    
    Random_bidder = random_bidder(c_cost_Random, d_cost_Random, K)
    
#     player_final_dists = []
    for run in tqdm(range(num_runs)):
        Random_bidder.restart()
        game_data = auction_data()
        for t in range(T):
            action, ind = Random_bidder.choose_action()
            
            Random_bidder.played_action = action
            Random_bidder.history_action.append(ind)
            bids = Trustful_profile + [action]
            x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)
            
            payoff = []
            for i in range(N):
                if i == N-1:
                    payoff_Random = payments[-1] - (0.5 * Random_bidder.cost[0] * x[-1] + Random_bidder.cost[1]) * x[-1]
                    Random_bidder.history_payoff.append(payoff_Random)
                    payoff.append(payoff_Random)
#                     if t==T-1:
#                         print(f'Run {run}) Random player payoff at round {t+1}: {payoff_Random}')
                else:
                    payoff_bidder = payments[i] - (0.5 * other_costs[i][0] * x[i] + other_costs[i][1]) * x[i]
                    payoff.append(payoff_bidder)
            game_data.payoffs.append(payoff)


            
            bidder = Random_bidder
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
## Case c ==> Hedge vs Hedge (All play Hedge)
# Case c ==> Hedge vs Hedge 

def all_Hedge(num_games, num_runs, T, file_name):
    types = ['All Hedge']
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
    types.append('All Hedge')
    #types.append('Random')

    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
            other_bidders.append(Hedge_bidder(c_list[i], d_list[i], K, max_payoff, T, has_seed = True))
            
        for type_idx, bidder_type in enumerate(types):
            if bidder_type == 'All Hedge':
                bidders = other_bidders + [Hedge_bidder(c_list[-1], d_list[-1], K, max_payoff, T)]
#             if bidder_type == 'Random':
#                 bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
                
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True, regret_all=True))
                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
#         pickle.dump(player_final_dists, file)

   ## Case d ==> Hedge vs Random (Others play Hedge vs B5 plays Random)

# Case c ==> Hedge vs Hedge 

def Hedge_vs_Random(num_games, num_runs, T, file_name):

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
    #types.append('All Hedge')
    types.append('Hedge vs Random')

    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
            other_bidders.append(Hedge_bidder(c_list[i], d_list[i], K, max_payoff, T))
            
        for type_idx, bidder_type in enumerate(types):
#             if bidder_type == 'All Hedge':
#                 bidders = other_bidders + [Hedge_bidder(c_list[-1], d_list[-1], K, max_payoff, T)]
            if bidder_type == 'Hedge vs Random':
                bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
                
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True, regret_all=True))
                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)

    ## Case e ==> Random vs Hedge (others play Random and B5 plays Hedge)

# Case e ==> Random vs Hedge 

def Random_vs_Hedge(num_games, num_runs, T, file_name):
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
    types.append('Random vs Hedge')
#     types.append('Random vs Random')


    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
           # other_bidders.append(random_bidder(c_list[i], d_list[i], K, has_seed = True))
            other_bidders.append(random_bidder(c_list[i], d_list[i], K))
            
        for type_idx, bidder_type in enumerate(types):
            if bidder_type == 'Random vs Hedge':
                bidders = other_bidders + [Hedge_bidder(c_list[-1], d_list[-1], K, max_payoff, T)]
#             if bidder_type == 'Random':
#                 bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
            
                
            player_final_dists = []
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True))

                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
#         pickle.dump(player_final_dists, file)

        
## Case f ==> Random vs Random (All play Random)
# Case f ==> Random vs Random 

def Random_vs_Random(num_games, num_runs, T, file_name):
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
    types.append('Random vs Random')
#     types.append('Random')


    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
            #other_bidders.append(random_bidder(c_list[i], d_list[i], K, has_seed = True))
            other_bidders.append(random_bidder(c_list[i], d_list[i], K))
            
        for type_idx, bidder_type in enumerate(types):
            if bidder_type == 'Random vs Random':
#                 bidders = other_bidders + [Hedge_bidder(c_list[-1], d_list[-1], K, max_payoff, T)]
#             if bidder_type == 'Random':
                bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
            
                
            player_final_dists = []
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True))

                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)


Trustful_vs_Hedge(num_games = 1, num_runs = 15, T = 200, file_name='TrustfulHG')
Trustful_vs_Random(num_games = 1, num_runs = 15, T = 200, file_name ='TrustfulRandom')
all_Hedge(num_games = 1, num_runs = 15, T = 200, file_name ='allHG')
Hedge_vs_Random(num_games = 1, num_runs = 15, T = 200, file_name = 'Hedge_vs_Random')
Random_vs_Hedge(num_games = 1, num_runs=15, T=200, file_name='Random_Hedge')
Random_vs_Random(num_games = 1, num_runs = 15, T=200, file_name='all_Random')
