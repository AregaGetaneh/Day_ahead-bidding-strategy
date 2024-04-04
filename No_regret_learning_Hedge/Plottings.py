import numpy as np
import cvxpy as cp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
import pickle
import re

def get_bidder_configuration():
    bidder_types = ['Trustful vs Hedge', 'Trustful vs Random', 'All Hedge', 'Hedge vs Random', 'Random vs Hedge', 'Random vs Random']
    bidder_colors = {
        'Trustful vs Hedge': 'green',
        'Trustful vs Random': 'brown',
        'All Hedge': 'blue', 
        'Hedge vs Random': 'gray', 
        'Random vs Hedge': 'red', 
        'Random vs Random': 'orange',
    }
    legend_labels = {
        'Trustful vs Hedge': 'Trustful vs Hedge',
        'Trustful vs Random': 'Trustful vs Random',
        'All Hedge': 'Hedge vs Hedge', 
        'Hedge vs Random': 'Hedge vs Random', 
        'Random vs Hedge': 'Random vs Hedge', 
        'Random vs Random': 'Random vs Random'
    }
    file_names = ['TrustfulHG', 'TrustfulRandom', 'allHG', 'Hedge_vs_Random', 'Random_Hedge', 'all_Random']
    
    return bidder_types, bidder_colors, legend_labels, file_names

def plot_regret(file_name):
    bidder_types, bidder_colors, legend_labels, file_names = get_bidder_configuration()
    
    markers = ['o', '*', 'x', '^', '4', '3', ">", '2', 'd']
    plt.rc("font", size=17)
    with open(f'{file_name}.pckl', 'rb') as file:
        global T
        T = pickle.load(file)
        types = pickle.load(file)
        game_data_profile = pickle.load(file)

    plot_from = 0
    idx_marker = 0
    for i, typ in enumerate(types):
        if typ in bidder_colors:
            data = np.array(
                [[game_data_profile[i][d].regrets[t][-1] for t in range(plot_from, T)] for d in range(len(game_data_profile[i]))])
            mean = np.mean(data, 0)
            std = np.std(data, 0)
            p = plt.plot(range(plot_from, T), mean, marker=markers[idx_marker], markevery=10, markersize=7, markerfacecolor='w', color=bidder_colors[typ])

            color = p[0].get_color()
            plt.fill_between(range(plot_from, T), mean - std, mean + std, alpha=0.1, color=color)

            plt.plot([], [], label=legend_labels.get(typ, typ), marker=markers[idx_marker], markersize=7, markerfacecolor='w', color=bidder_colors[typ])
            
            idx_marker += 1
plt.figure(figsize=(10, 4))

plot_regret('allHG')
plot_regret('Random_Hedge')
plot_regret('TrustfulHG')
plot_regret('TrustfulRandom')
plot_regret('Hedge_vs_Random')
plot_regret('all_Random')

legend = plt.legend(loc='upper center', bbox_to_anchor=(0.54, 0.93), bbox_transform=plt.gcf().transFigure, ncol=2)
legend.get_frame().set_alpha(0)

plt.xlabel('Time (rounds)')
plt.xlim([0, T])
plt.ylim([0, 12000])
plt.ylabel('Regret [€]')

plt.tight_layout()
plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plt.savefig('regret_combined.pdf', bbox_inches="tight")
plt.show()

bidder_types = ['Trustful vs Hedge','Trustful vs Random','All Hedge','Hedge vs Random', 'Random vs Hedge', 'Random vs Random']
bidder_colors = {'Trustful vs Hedge': 'green','Trustful vs Random': 'brown','All Hedge': 'blue', 'Hedge vs Random': 'gray', 'Random vs Hedge': 'red', 'Random vs Random': 'orange', }
legend_labels = {'Trustful vs Hedge': 'Trustful vs Hedge','Trustful vs Random': 'Trustful vs Random',
                 'All Hedge': 'Hedge vs Hedge', 'Hedge': 'Random vs Hedge', 'Random vs Random': 'Random vs Random'} 
file_names = ['TrustfulHG', 'TrustfulRandom', 'allHG', 'Hedge_vs_Random', 'Random_Hedge', 'all_Random']

markers = ['o', '*', 'x', '^', '4', '3', ">", '2', 'd']

def collect_payoff_5(file_names, Diag, Trustful, bidder_colors, bidder_types):
    
    bidder_types, bidder_colors, legend_labels, file_names = get_bidder_configuration()
    
    plt.rc("font", size=17)
    plt.figure(figsize=(10, 4))


    for file_name in file_names:
        with open(f'{file_name}.pckl', 'rb') as file:
            T = pickle.load(file)
            types = pickle.load(file)
            game_data_profile = pickle.load(file)

        plot_from = 0
        idx_marker = 0

        mean_payoffs = []  # Changed variable name to mean_payoffs

        for i, typ in enumerate(types):
            data_non_average = np.array(
            [[game_data_profile[i][d].payoffs[t][-1] for t in range(plot_from, T)] for d in range(len(game_data_profile[i]))])
            data = [[sum(sublist[:i+1]) / (i+1) for i in range(len(sublist))] for sublist in data_non_average]
            mean = np.mean(data, 0)
            std = np.std(data, 0)

            p = plt.plot(range(plot_from, T), mean, marker='o', markevery=10, markersize=7,  markerfacecolor='w', color=bidder_colors[typ], label=legend_labels.get(typ, typ))
            color = p[0].get_color()
            plt.fill_between(range(plot_from, T), mean - std,
                             mean + std, alpha=0.1,
                             color=color)

        idx_marker += 1
    plt.axhline(y = Diag, color='#069AF3', linestyle='--', label=f'Best Response')
    plt.axhline(y = Trustful, color='#F96306', linestyle='--', label=f'Trustful')
    



    plt.xlabel('Time (rounds)')
    plt.xlim([0, T])
    plt.ylim([0, 20000])
    plt.ylabel('Payoff [€]') 
    plt.tight_layout()
    plt.ticklabel_format(style = "sci", axis = "y", scilimits = (0,0))
    plt.savefig(f'payoff_bidder_5.pdf', bbox_inches="tight")

Diag = 3232.21  
Trustful = 1118.64


collect_payoff_5(file_names, Diag, Trustful, bidder_colors, bidder_types)

markers = ['o', '*', 'x', '^', '4', '3', ">", '2', 'd']
def plot_combined_SW(file_names, Diag, Trustful, bidder_colors, bidder_types):
    
    bidder_types, bidder_colors, legend_labels, file_names = get_bidder_configuration()
    plt.rc("font", size= 17)
    plt.figure(figsize=(10, 4))


    for file_name in file_names:
        with open(f'{file_name}.pckl', 'rb') as file:
            T = pickle.load(file)
            types = pickle.load(file)
            game_data_profile = pickle.load(file)

        plot_from = 0
        idx_marker = 0

        for i, typ in enumerate(types):
            if typ in bidder_types:
                data_non_average = np.array(
                    [[game_data_profile[i][d].SW[t] for t in range(plot_from, T)]
                     for d in range(len(game_data_profile[i]))])
                
                data = [[sum(sublist[:i + 1]) / (i + 1) for i in range(len(sublist))] for sublist in data_non_average]
                mean = np.mean(data, 0)
                #print(mean)
                std = np.std(data, 0)

                if idx_marker < len(markers):
                    player_marker = markers[idx_marker]
                else:
                    player_marker = 'o' 

                p = plt.plot(range(plot_from, T), mean, marker=player_marker, markevery=10, markersize=7,
                             markerfacecolor='w', color=bidder_colors[typ], label=legend_labels.get(typ, typ))

                color = p[0].get_color()
                plt.fill_between(range(plot_from, T), mean - std,
                                 mean + std, alpha=0.1,
                                 color=color)

                idx_marker += 1
    

    plt.axhline(y= Diag, color='#069AF3', linestyle='--', label=f'Best Response')
    plt.axhline(y= Trustful, color='#F96306', linestyle='--', label=f'Trustful')
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.55, 1.11), bbox_transform=plt.gcf().transFigure, ncol=2)
        
    legend.get_frame().set_alpha(0)

    plt.xlabel('Time (rounds)')
    plt.xlim([0, T])
    plt.ylim([0, 50000])
    plt.ylabel('Social cost [€]')

    plt.tight_layout()
    plt.ticklabel_format(style = "sci", axis = "y", scilimits = (0,0))
    plt.savefig(f'Social_welfare_combined.pdf', bbox_inches="tight")
    plt.show()

Diag = 24408.55  
Trustful = 19418.91

plot_combined_SW(file_names, Diag, Trustful, bidder_colors, bidder_types)


def plot_combined_MCP(file_names, Diag, Trustful, bidder_colors, bidder_types):
    
    bidder_types, bidder_colors, legend_labels, file_names = get_bidder_configuration()
    plt.rc("font", size = 17)
    plt.figure(figsize=(10, 4))

    for file_name in file_names:
        with open(f'{file_name}.pckl', 'rb') as file:
            T = pickle.load(file)
            types = pickle.load(file)
            game_data_profile = pickle.load(file)
            
        plot_from = 0
        idx_marker = 0

        for i, typ in enumerate(types):
            if typ in bidder_types:
                data_non_average = np.array(
                    [[game_data_profile[i][d].marginal_prices[t] for t in range(plot_from, T)] for d in range(len(game_data_profile[i]))])
                data = [[sum(sublist[:i + 1]) / (i + 1) for i in range(len(sublist))] for sublist in data_non_average]
                mean = np.mean(data, 0)
                std = np.std(data, 0)

              
                if idx_marker < len(markers):
                    player_marker = markers[idx_marker]
                else:
                    player_marker = 'o' 

                p = plt.plot(range(plot_from, T), mean, marker=player_marker, markevery=10, markersize=7,
                             markerfacecolor='w', color=bidder_colors[typ], label=legend_labels.get(typ, typ))

                color = p[0].get_color()
                plt.fill_between(range(plot_from, T), mean - std,
                                 mean + std, alpha=0.1,
                                 color=color)

                idx_marker += 1
    
   
    plt.axhline(y = Diag, color='#069AF3', linestyle='--', label=f'Diag')
    plt.axhline(y = Trustful, color='#F96306', linestyle='--', label=f'Trustful')



    plt.xlabel('Time (rounds)')
    plt.xlim([plot_from, 200])
    plt.ylim([0, 50])
    plt.ylabel('Clearing price [€/MWh]')

    plt.tight_layout()
    plt.savefig(f'market_clearning_prices_combined.pdf', bbox_inches="tight")
    plt.show()


Diag = 20.64  
Trustful = 15.73
 

plot_combined_MCP(file_names, Diag, Trustful, bidder_colors, bidder_types)

                
