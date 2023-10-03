import matplotlib.pyplot as plt
import pickle
import pickle

# For all plots

bidder_colors = {
    'GPMW 0.7': 'red',
    'All GPMW': 'blue',
    'GPMW vs BR': 'green'
}
T = None



def plot_regret(file_name, bidder_colors, label):
    
    legend_labels = {
    'GPMW 0.7': 'Random',
    'All GPMW': 'GPMW',
    'GPMW vs BR': 'BR'
    }
    
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
            plt.fill_between(range(plot_from, T), mean - std,
                             mean + std, alpha=0.1,
                             color=color)
            
           
            plt.plot([], [], label=legend_labels.get(typ, typ), marker=markers[idx_marker], markersize=7, markerfacecolor='w', color=bidder_colors[typ])
            
            idx_marker += 1

plt.figure(figsize=(9, 4))
plot_regret('resGP', bidder_colors, label=0)
plot_regret('resGPBR', bidder_colors, label=1)
plot_regret('res', bidder_colors, label=2)
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.53, 1.0), bbox_transform=plt.gcf().transFigure, ncol=4)
legend.get_frame().set_alpha(0)
plt.xlabel('Time (rounds)')
plt.xlim([0, T])  
plt.ylim([0, 12000])
plt.ylabel('Regret [€]')
plt.tight_layout()
plt.ticklabel_format(style = "sci", axis = "y", scilimits = (0,0))
plt.savefig('regret_combined.pdf', bbox_inches="tight")
plt.show()



def plot_combined_SW(file_names,  diagonalization, trustful, bidder_colors, bidder_types):
    markers = ['o', '*', 'x']
    plt.rc("font", size=17)
    plt.figure(figsize=(9, 4))

    legend_labels = {
        'GPMW 0.7': 'Random',
        'All GPMW': 'GPMW',
        'GPMW vs BR': 'BR'
    } 

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
                    [[game_data_profile[i][d].SW[t] for t in range(plot_from, T)] for d in range(len(game_data_profile[i]))])
                data = [[sum(sublist[:i + 1]) / (i + 1) for i in range(len(sublist))] for sublist in data_non_average]
                mean = np.mean(data, 0)
                std = np.std(data, 0)

                # Use different markers for players
                if idx_marker < len(markers):
                    player_marker = markers[idx_marker]
                else:
                    player_marker = 'o'  # Use a default marker for additional players

                p = plt.plot(range(plot_from, T), mean, marker=player_marker, markevery=10, markersize=7,
                             markerfacecolor='w', color=bidder_colors[typ], label=legend_labels.get(typ, typ))

                color = p[0].get_color()
                plt.fill_between(range(plot_from, T), mean - std,
                                 mean + std, alpha=0.1,
                                 color=color)

                idx_marker += 1

    plt.axhline(y= diagonalization, color='#069AF3', linestyle='--', label=f'Diag')
    plt.axhline(y= trustful, color='#F96306', linestyle='--', label=f'Trustful')

    
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.54, 1.03), bbox_transform=plt.gcf().transFigure, ncol=3)
    legend.get_frame().set_alpha(0)
    plt.xlabel('Time (rounds)')
    plt.xlim([0, T])
    plt.ylim([0, 50000])
    plt.ylabel('Social cost [€]')
    plt.tight_layout()
    plt.ticklabel_format(style = "sci", axis = "y", scilimits = (0,0))
    plt.savefig(f'Social_welfare_combined.pdf', bbox_inches="tight")
    plt.show()


file_names = ['resGP', 'resGPBR', 'res']
diagonalization = 24408.55  
trustful = 19418.91
bidder_types = ['All GPMW', 'GPMW vs BR', 'GPMW 0.7']  

# Call the function to plot 
plot_combined_SW(file_names,  diagonalization, trustful, bidder_colors, bidder_types)



def collect_payoff_Bidder_1(file_names, diagonalization, trustful, bidder_colors, bidder_types):
    plt.rc("font", size=17)
    plt.figure(figsize=(9, 4))

    legend_labels = {
        'GPMW 0.7': 'Random',
        'All GPMW': 'GPMW',
        'GPMW vs BR': 'BR'
    } 

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
            [[game_data_profile[i][d].payoffs[t][0] for t in range(plot_from, T)] for d in range(len(game_data_profile[i]))])
        data = [[sum(sublist[:i+1]) / (i+1) for i in range(len(sublist))] for sublist in data_non_average]
        mean = np.mean(data, 0)
        std = np.std(data, 0)

        p = plt.plot(range(plot_from, T), mean, marker='o', markevery=10, markersize=7,  markerfacecolor='w', color=bidder_colors[typ], label=legend_labels.get(typ, typ))
        color = p[0].get_color()
        plt.fill_between(range(plot_from, T), mean - std,
                         mean + std, alpha=0.1,
                         color=color)

        idx_marker += 1
    
    plt.axhline(y= diagonalization, color='#069AF3', linestyle='--', label=f'Diag')
    plt.axhline(y= trustful, color='#F96306', linestyle='--', label=f'Trustful')

    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.54, 1.03), bbox_transform=plt.gcf().transFigure, ncol=3) 
    legend.get_frame().set_alpha(0)
    plt.xlabel('Time (rounds)')
    plt.xlim([0, T])
    plt.ylim([0, 16000])
    plt.ylabel('Payoff [€]')  # Changed label to 'Payoff'
    plt.tight_layout()
    plt.ticklabel_format(style = "sci", axis = "y", scilimits = (0,0))
    plt.savefig(f'payoff_bidder_1.pdf', bbox_inches="tight")


file_names = ['resGP', 'resGPBR', 'res']
diagonalization = 2158.42  
trustful = 2055.66
bidder_types = ['All GPMW', 'GPMW vs BR', 'GPMW 0.7']  

collect_payoff_Bidder_1(file_names, diagonalization, trustful, bidder_colors, bidder_types) 



def collect_payoff_Bidder_5(file_names, diagonalization, trustful, bidder_colors, bidder_types):
    plt.rc("font", size=17)
    plt.figure(figsize=(9, 4))

    legend_labels = {
        'GPMW 0.7': 'Random',
        'All GPMW': 'GPMW',
        'GPMW vs BR': 'BR'
    }  

    for file_name in file_names:
        with open(f'{file_name}.pckl', 'rb') as file:
            T = pickle.load(file)
            types = pickle.load(file)
            game_data_profile = pickle.load(file)

        plot_from = 0
        idx_marker = 0

        mean_payoffs = [] 

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

    plt.axhline(y= diagonalization, color ='#069AF3', linestyle ='--', label = f'Diag')
    plt.axhline(y= trustful, color='#F96306', linestyle ='--', label = f'Trustful')

    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.54, 1.03), bbox_transform=plt.gcf().transFigure, ncol=3)
    legend.get_frame().set_alpha(0)
    plt.xlabel('Time (rounds)')
    plt.xlim([0, T])
    plt.ylim([0, 16000])
    plt.ylabel('Payoff [€]')  
    plt.tight_layout()
    plt.ticklabel_format(style = "sci", axis = "y", scilimits = (0,0))
    plt.savefig(f'payoff_bidder_5.pdf', bbox_inches="tight")


file_names = ['resGP', 'resGPBR', 'res']
diagonalization = 12741.34  
trustful = 5755.82
bidder_types = ['All GPMW', 'GPMW vs BR', 'GPMW 0.7'] 

collect_payoff_Bidder_5(file_names, diagonalization, trustful, bidder_colors, bidder_types)                


# Market clearing price



def plot_combined_MCP(file_names, diagonalization, trustful, bidder_colors, bidder_types):
    plt.rc("font", size = 17)
    plt.figure(figsize = (9, 4))
    markers = ['o', '*', 'x', '^']

    legend_labels = {
        'GPMW 0.7': 'Random',
        'All GPMW': 'GPMW',
        'GPMW vs BR': 'BR'
    }
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
                             markerfacecolor ='w', color=bidder_colors[typ], label=legend_labels.get(typ, typ))

                color = p[0].get_color()
                plt.fill_between(range(plot_from, T), mean - std,
                                 mean + std, alpha = 0.1,
                                 color = color)

                idx_marker += 1
    
   
    plt.axhline(y = diagonalization, color='#069AF3', linestyle= '--', label= f'Diag')
    plt.axhline(y = trustful, color='#F96306', linestyle = '--', label = f'Trustful')

    legend = plt.legend(loc ='upper center', bbox_to_anchor = (0.54, 1.03), bbox_transform = plt.gcf().transFigure, ncol = 4) 
    legend.get_frame().set_alpha(0)
    plt.xlabel('Time (rounds)')
    plt.xlim([0, T])
    plt.ylim([0, 50])
    plt.ylabel('Clearing price [€/MWh]')
    plt.tight_layout()
    plt.savefig(f'market_clearning_prices_combined.pdf', bbox_inches="tight")
    plt.show()

file_names = ['resGP', 'resGPBR', 'res']
diagonalization = 20.64  
trustful = 15.73
bidder_types = ['All GPMW', 'GPMW vs BR', 'GPMW 0.7']  

# Call the function to plot
plot_combined_MCP(file_names, diagonalization, trustful, bidder_colors, bidder_types)


# Plotting Bid prices for Bidder 1 and 5 (c and d values)


def collect_bid_price_1_c(file_names, diagonalization, trustful, bidder_colors, bidder_types):
# def collect_bid_price_5_c(file_names, diagonalization, trustful, bidder_colors, bidder_types):

    plt.rc("font", size=16)
    plt.figure(figsize=(9, 4))
    legend_labels = {
        'GPMW 0.7': 'Random',
        'All GPMW': 'GPMW',
        'GPMW vs BR': 'BR'
    } 


    for file_name in file_names:
        with open(f'{file_name}.pckl', 'rb') as file:
            T = pickle.load(file)
            types = pickle.load(file)
            game_data_profile = pickle.load(file)

        plot_from = 0
        idx_marker = 0

        mean_bid_prices = []

        for i, typ in enumerate(types):
            bids_c = np.array(
                [[game_data_profile[i][d].bids[t][0][0] for t in range(plot_from, T)] for d in
                 range(len(game_data_profile[i]))])

            
            data = np.multiply(bids_c, 1)
            data_non_average = data 
            data = [[sum(sublist[:j + 1]) / (j + 1) for j in range(len(sublist))] for sublist in data_non_average]
            mean_bid_prices.append(np.mean(data, axis=0))

        mean_bid_prices = np.mean(mean_bid_prices, axis=0)
        std_bid_prices = np.std(mean_bid_prices, axis=0)
        p = plt.plot(range(plot_from, T), mean_bid_prices, marker='o', markevery=10, markersize=7,
                     markerfacecolor='w', color=bidder_colors[typ], label=legend_labels.get(typ, typ))
        color = p[0].get_color()
        plt.fill_between(range(plot_from, T), mean_bid_prices - std_bid_prices,
                         mean_bid_prices + std_bid_prices, alpha=0.1,
                             color=color)

        idx_marker += 1
  
    plt.axhline(y= diagonalization, color='#069AF3', linestyle='--', label=f'Diag')
    plt.axhline(y= trustful, color='#F96306', linestyle='--', label=f'Trustful')

   
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.54, 1.03), bbox_transform=plt.gcf().transFigure, ncol=4)
        
    legend.get_frame().set_alpha(0)

    plt.xlabel('Time (rounds)')
    plt.xlim([0, T])
    plt.ylabel('c values [€/MWh$^2]$')

    plt.tight_layout()
    plt.savefig(f'bid_price_bidder_1_C_values.pdf', bbox_inches="tight")
#     plt.savefig(f'bid_price_bidder_1_C_values.pdf', bbox_inches="tight") # Uncomment for Bidder 5
    plt.show()


file_names = ['resGP', 'resGPBR', 'res']
diagonalization = 0.095  
trustful = 0.07

'''Uncomment the code below for Bidder 5''' 

# diagonalization = 0.02 
# trustful = 0.01

bidder_types = ['All GPMW', 'GPMW vs BR', 'GPMW 0.7']  


collect_bid_price_1_c(file_names, diagonalization, trustful, bidder_colors, bidder_types)

#Change the index to -1 to plot Bidder 5's values to extract bid_c, i.e game_data_profile[i][d].bids[t][-1][0] 
#and uncomment the call bellow''' 


# collect_bid_price_5_c(file_names, diagonalization, trustful, bidder_colors, bidder_types)
                

def collect_bid_price_1_d(file_names, diagonalization, trustful, bidder_colors, bidder_types):
# def collect_bid_price_5_d(file_names, diagonalization, trustful, bidder_colors, bidder_types):

    plt.rc("font", size=17)
    plt.figure(figsize=(9, 4))
    legend_labels = {
        'GPMW 0.7': 'Random',
        'All GPMW': 'GPMW',
        'GPMW vs BR': 'BR'
    }

    for file_name in file_names:
        with open(f'{file_name}.pckl', 'rb') as file:
            T = pickle.load(file)
            types = pickle.load(file)
            game_data_profile = pickle.load(file)

        plot_from = 0
        idx_marker = 0

        mean_bid_prices = []

        for i, typ in enumerate(types):
            bids_d = np.array(
                [[game_data_profile[i][d].bids[t][0][1] for t in range(plot_from, T)] for d in
                 range(len(game_data_profile[i]))])
            
            data = np.multiply(bids_d, 1)
            data_non_average = data 
            data = [[sum(sublist[:j + 1]) / (j + 1) for j in range(len(sublist))] for sublist in data_non_average]
            mean_bid_prices.append(np.mean(data, axis=0))

        mean_bid_prices = np.mean(mean_bid_prices, axis=0)
        std_bid_prices = np.std(mean_bid_prices, axis=0)
        p = plt.plot(range(plot_from, T), mean_bid_prices, marker='o', markevery=10, markersize=7,
                     markerfacecolor='w', color=bidder_colors[typ], label=legend_labels.get(typ, typ))
        color = p[0].get_color()
        plt.fill_between(range(plot_from, T), mean_bid_prices - std_bid_prices,
                         mean_bid_prices + std_bid_prices, alpha=0.1,
                             color=color)

        idx_marker += 1
    plt.axhline(y= diagonalization, color='#069AF3', linestyle='--', label=f'Diag')
    plt.axhline(y= trustful, color='#F96306', linestyle='--', label=f'Trustful')

    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.54, 1.03), bbox_transform=plt.gcf().transFigure, ncol=3) 
    legend.get_frame().set_alpha(0)
    plt.xlabel('Time (rounds)')
    plt.xlim([0, T])
    plt.ylabel('d values [€/MWh]')

    plt.tight_layout()
    plt.savefig(f'bid_price_bidder_1_d_values.pdf', bbox_inches="tight")
#     plt.savefig(f'bid_price_bidder_5_d_values.pdf', bbox_inches="tight") # Uncomment this for Bidder 5 
    plt.show()


file_names = ['resGP', 'resGPBR', 'res']
diagonalization = 13.0  
trustful = 9.0
bidder_types = ['All GPMW', 'GPMW vs BR', 'GPMW 0.7'] 

#collect_bid_price_1_d(file_names, diagonalization, trustful, bidder_colors, bidder_types)

#Change the index to -1 to plot Bidder 5's values to extract bid_d, i.e game_data_profile[i][d].bids[t][-1][1] 
#and uncomment the call bellow'''

# collect_bid_price_5_d(file_names, diagonalization, trustful, bidder_colors, bidder_types)     
