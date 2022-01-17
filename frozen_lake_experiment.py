# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:14:34 2020

@author: nicholas.park
"""

from hiive.mdptoolbox.mdp import *
from hiive.visualization import *

import numpy as np
import gym
from gym import wrappers
import time
import matplotlib.pyplot as plt
import pandas as pd
from frozen_lake import FrozenLakeEnv
from helpers import visualize_policy, visualize_value, visualize_env
from constants import design_random_map
from policy_iteration import policy_iteration
from value_iteration import value_iteration
from q_learning import q_learning
import plotting as plotting

plt.style.use("default")
def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] != 'H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]

def return_term_goal_name(randommap):
    x = list(''.join(randommap))
    n = int(np.sqrt(len(x)))
    name = "{}x{}".format(n,n)
    t = []
    g = []
    for i,e in enumerate(x):
        if e == "H":
            t.append(i)
        elif e == "G":
            g.append(i)
    term = {}
    goal = {}
    term[name] = t
    goal[name] = g
    return term, goal, name



def pymdp_PR(env):
    P = []
    R = []
    for a in range(env.nA):
        P_a = np.zeros([env.nS, env.nS])
        R_a = np.zeros([env.nS, env.nS])
        for s in range(env.nS):
            P_sa = []     
            for i in range(len(env.P[s][a])): #list of tuples
                P_sa.append(list(env.P[s][a][i]))
            temp1 = pd.DataFrame(np.array(P_sa)[:,:2], columns = ["probs", "newstate"]).groupby("newstate")["probs"].sum().reset_index()# list of tuple
            temp2 = pd.DataFrame(np.array(P_sa)[:,1:3], columns = ["newstate", "reward"]).groupby("newstate")["reward"].mean().reset_index()# list of tuple
            for case1 in temp1.values:
                P_a[s, int(case1[0])]=case1[1]
            for case2 in temp2.values:
                R_a[s, int(case2[0])]=case2[1]
        P.append(P_a)
        R.append(R_a)
    return P,R


# <visualize random maps>

# runtimes = []
# iters = []
# tested_size = np.arange(1,7)*6
# tested_sliprates = np.linspace(0.001,0.999,num=10)

# for n in tested_size:
    
#     runtimes_sliprate=  []
#     iters_sliprate =  []
    
#     for sliprate in tested_sliprates:
        
#         prob = 0.8
#         gamma = 0.9
#         theta = 0.0001
#         rewards = (-0.01, -1, 1)
#         name = "{}x{}".format(n,n)
#         randommap = generate_random_map(size = n, p= prob)  # creates dictionary
#         term, goal, name = return_term_goal_name(randommap)
#         # print(name)
#         env_kwargs = {
#             'desc': randommap,
#             'slip_rate': sliprate,
#             'rewards': rewards
#         }
#         env = FrozenLakeEnv(**env_kwargs)
#         pi_env = env.unwrapped
#         # visualize_env(env, term, goal, name, title = name + " environment visualization")
#         # plt.show()
#         # pi_policy, pi_V, n_iter, runtime = policy_iteration(pi_env, discount_factor=gamma, theta=theta)
#         # print(pi_policy)
#         P = pymdp_PR(pi_env)[0]
#         R = pymdp_PR(pi_env)[1]
#         pi = PolicyIterationModified(P, R, gamma, epsilon = theta, )
#         pi.run()
#         # print(pi.iter)
#         # print(pi.time)
#         n_iter = pi.iter
#         runtime = pi.time
#         iters_sliprate.append(n_iter)
#         runtimes_sliprate.append(runtime)

#     runtimes.append(runtimes_sliprate)
#     iters.append(iters_sliprate)

# # print(runtimes_sliprate)

# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("Policy Iter.: Total number of iterations")
# plt.ylabel("Number of iteration")
# plt.xlabel("Slip rate")
# plt.grid(linewidth = 0.2)    
# for i, iters_sliprate in enumerate(iters):
#     plt.plot(tested_sliprates, iters_sliprate, label="gridsize: {}x{}".format(int(tested_size[i]), int(tested_size[i])))
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Policy Iter.: runtime")
# plt.ylabel("Runtimne")
# plt.xlabel("Slip rate")
# plt.grid(linewidth = 0.2)    
# for i, runtimes_sliprate in enumerate(runtimes):
#     plt.plot(tested_sliprates,runtimes_sliprate, label="gridsize: {}x{}".format(int(tested_size[i]), int(tested_size[i])))
# plt.legend()
# plt.savefig("sliprate experiment2 (PI)")



# tested_size = [5, 10, 20] 
# tested_sliprates = np.linspace(0.001,0.999,num=10)
# tested_seeds = np.arange(1,11)

# match_rate_per_size_policy = []
# match_rate_per_size_v = []

# for n in tested_size:

#     total_policy_match= []
#     total_value_match= []

#     for seed in tested_seeds:

#         policy_match = []
#         value_match = []
        
#         for sliprate in tested_sliprates:
            

#             prob = 0.8
#             gamma = 0.9
#             theta = 0.0001
#             rewards = (-0.01, -1, 1)
#             name = "{}x{}".format(n,n)
#             randommap = generate_random_map(size = n, p= prob)  # creates dictionary
#             term, goal, name = return_term_goal_name(randommap)
#             # print(name)
#             env_kwargs = {
#                 'desc': randommap,
#                 'slip_rate': sliprate,
#                 'rewards': rewards
#             }
#             env = FrozenLakeEnv(**env_kwargs)
#             pi_env = env.unwrapped
#             P = pymdp_PR(pi_env)[0]
#             R = pymdp_PR(pi_env)[1]
#             pi = PolicyIterationModified(P, R, gamma = gamma, epsilon=0.005)
#             vi_env = env.unwrapped
#             vi = ValueIteration(P, R, gamma =gamma, epsilon= 0.005)
#             pi.run()
#             vi.run()
#             policy_match.append( int(np.all(np.isclose(pi.policy, vi.policy, atol=0.00001)) ))
#             value_match.append(int(np.all(np.isclose(pi.V, vi.V, atol=0.00001))  ))
#         print(policy_match)
            
#         total_policy_match.append(policy_match)
#         total_value_match.append(value_match)
#         # print(total_policy_match)

#     policy_match_rate = np.sum(total_policy_match, axis = 0)/len(tested_seeds)
#     # print(policy_match_rate)
#     value_match_rate = np.sum(total_value_match, axis = 0)/len(tested_seeds)
#     match_rate_per_size_policy.append(policy_match_rate)
#     match_rate_per_size_v.append(value_match_rate)


# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("PI and VI: Policy Match Rate (atol = 0.00001)")
# plt.ylabel("Match Rate")
# plt.xlabel("Slip rate")
# plt.grid(linewidth = 0.2)    
# for i, e in enumerate(match_rate_per_size_policy):
#     plt.plot(tested_sliprates, e, label="gridsize: {}x{}".format(int(tested_size[i]), int(tested_size[i])))
# plt.ylim(0, 1.05)
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("PI and VI: Value Match Rate (atol = 0.00001)")
# plt.ylabel("Match Rate")
# plt.xlabel("Slip rate")
# plt.grid(linewidth = 0.2)    
# for i, e in enumerate(match_rate_per_size_v):
#     plt.plot(tested_sliprates, e, label="gridsize: {}x{}".format(int(tested_size[i]), int(tested_size[i])))
# plt.ylim(0, 1.05)
# plt.legend()
# plt.savefig("PI_VI_matchrate_experiment_2")



# n = 10
# prob = 0.8
# gamma = 0.9
# theta = 0.0001
# rewards = (-0.01, -1, 1)
# name = "{}x{}".format(n,n)
# randommap = generate_random_map(size = n, p= prob)  # creates dictionary
# term, goal, name = return_term_goal_name(randommap)
# # print(name)
# env_kwargs = {
#     'desc': randommap,
#     'slip_rate': 0.2,
#     'rewards': rewards
# }
# env = FrozenLakeEnv(**env_kwargs)
# pi_env = env.unwrapped
# P = pymdp_PR(pi_env)[0]
# R = pymdp_PR(pi_env)[1]
# pi = QLearning(P, R, gamma = gamma, )
# pi.run()
# print(pi.rt_per_iter)






def plot_epsilon_decay_geom(epsilon, decay, n_episodes, stats):
    e_prime = epsilon * np.ones(n_episodes)
    for i in range(n_episodes):
        e_prime[i] *= decay ** i
    smoothing_window=10
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed, label='Episode reward')
    plt.plot(e_prime, label='epsilon', linestyle='--')
    plt.title("Epsilon-greedy with decay (epsilon=1.0, decay=0.999)")
    plt.xlabel('Episode')
    plt.legend(loc='best')
    plt.show()
    
def plot_epsilon_decay_arith(epsilon, decay, n_episodes, stats):
    e_prime = epsilon * np.ones(n_episodes)
    for i in range(n_episodes):
        e_prime[i] -= decay*i
    smoothing_window=10
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed, label='Episode reward')
    plt.plot(e_prime, label='epsilon', linestyle='--')
    plt.title("Epsilon-greedy with decay (epsilon=1.0, decay=0.999)")
    plt.xlabel('Episode')
    plt.legend(loc='best')
    plt.show()
    

def get_state_action_value(final_policy):
    return np.max(final_policy, axis=1)






# runtimes = []
# iters = []
# tested_size = np.arange(1,7)*6
# tested_sliprates = np.linspace(0.001,0.999,num=30)

# for n in tested_size:
    
#     runtimes_sliprate=  []
#     iters_sliprate =  []
    
#     for sliprate in tested_sliprates:
        
#         prob = 0.8
#         gamma = 0.9
#         theta = 0.0001
#         rewards = (-0.01, -1, 1)
#         name = "{}x{}".format(n,n)
#         randommap = generate_random_map(size = n, p= prob)  # creates dictionary
#         term, goal, name = return_term_goal_name(randommap)
#         # print(name)
#         env_kwargs = {
#             'desc': randommap,
#             'slip_rate': sliprate,
#             'rewards': rewards
#         }
#         env = FrozenLakeEnv(**env_kwargs)
#         vi_env = env.unwrapped
#         # visualize_env(env, term, goal, name, title = name + " environment visualization")
#         # plt.show()
#         vi_policy, vi_V, n_iter, runtime = value_iteration(vi_env, discount_factor=gamma, theta=theta)
#         iters_sliprate.append(n_iter)
#         runtimes_sliprate.append(runtime)

#     runtimes.append(runtimes_sliprate)
#     iters.append(iters_sliprate)

# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("Value Iter.: Total number of iterations")
# plt.ylabel("Number of iteration")
# plt.xlabel("Slip rate")
# plt.grid(linewidth = 0.2)    
# for i, iters_sliprate in enumerate(iters):
#     plt.plot(tested_sliprates, iters_sliprate, label="gridsize: {}x{}".format(int(tested_size[i]), int(tested_size[i])))
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Value Iter.: runtime")
# plt.ylabel("Runtimne")
# plt.xlabel("Slip rate")
# plt.grid(linewidth = 0.2)    
# for i, runtimes_sliprate in enumerate(runtimes):
#     plt.plot(tested_sliprates,runtimes_sliprate, label="gridsize: {}x{}".format(int(tested_size[i]), int(tested_size[i])))
# plt.legend()
# plt.savefig("sliprate experiment1 (VI)")




# n = 10
# sliprate = 0.2
# prob = 0.8
# gamma = 0.9
# theta = 0.0001
# rewards = (-0.01, -1, 1)

# tested_size = [5, 10, 20] 

# err_vi = []
# err_pi = []
# err_q = []
# ri_vi = []
# ri_pi = []
# ri_q = []

# for n in tested_size:
    
#     name = "{}x{}".format(n,n)
#     randommap = generate_random_map(size = n, p= prob)  # creates dictionary
#     term, goal, name = return_term_goal_name(randommap)
#     env_kwargs = {
#         'desc': randommap,
#         'slip_rate': sliprate,
#         'rewards': rewards
#     }
#     env = FrozenLakeEnv(**env_kwargs)
#     pi_env = env.unwrapped
#     P = pymdp_PR(pi_env)[0]
#     R = pymdp_PR(pi_env)[1]
#     pi = PolicyIteration(P, R, gamma = gamma,)
#     vi = ValueIteration(P, R, gamma =gamma, )
#     # q = QLearning(P, R, gamma = gamma, n_iter = 50000)
#     pi.run()
#     vi.run()
#     # q.run()
    
#     err_pi.append(pi.err)
#     err_vi.append(vi.err)
#     ri_vi.append(vi.rt_per_iter)
#     ri_pi.append(pi.rt_per_iter)
#     # ri_q.append(q.rt_per_iter )

#     # utility_vi_perseed.append(vi.)


# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("Error Convergence")
# plt.ylabel("Error")
# plt.xlabel("completion rate (in iteration)")
# plt.grid(linewidth = 0.2)  
# for i,e in enumerate(tested_size):  
#     plt.plot(np.arange(1, len(err_pi[i]) +1 ) , err_pi[i], label="gridsize: {}x{}, policy iteration".format(e,e))
#     plt.plot(np.arange(1, len(err_vi[i]) +1 ) , err_vi[i], label="gridsize: {}x{}, value iteration".format(e,e))
#     # plt.plot(np.linspace(0, 1, num = len(utility_q[i])), utility_q[i], label="gridsize: {}x{}, q learning".format(e,e))
# plt.xlim(-2,50)
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Runtime per iteration")
# plt.ylabel("Runtime in seconds")
# plt.xlabel("completion rate (in iteration)")
# plt.grid(linewidth = 0.2)    
# for i,e in enumerate(tested_size):  
#     plt.plot(np.arange(1, len(ri_pi[i]) +1 ) , ri_pi[i], label="gridsize: {}x{}, policy iteration".format(e,e),linewidth = 1)
#     plt.plot(np.arange(1, len(ri_vi[i]) +1 ) , ri_vi[i], label="gridsize: {}x{}, value iteration".format(e,e),linewidth = 1)
#     print("The number of iteration: gridsize: {}x{}, pi: {}".format(e,e,len((ri_pi[i]))))
#     print("Average runtime per iteration: gridsize: {}x{}, pi: {}".format(e,e,np.mean(ri_pi[i])))    
#     print("The number of iteration: gridsize: {}x{}, vi: {}".format(e,e,len((ri_vi[i]))))
#     print("Average runtime per iteration: gridsize: {}x{}, vi: {}".format(e,e,np.mean(ri_vi[i])))
#     # plt.plot(np.linspace(0, 1, num = len(ri_q[i])), ri_q[i], label="gridsize: {}x{}, q learning".format(e,e), linewidth = 0.13)
# plt.legend() 
# plt.savefig("PI VI Q iteration convergence and runtime")



# Epsilon Decay



# n = 25
# sliprate = 0.2
# prob = 0.8
# gamma = 0.9
# theta = 0.0001
# rewards = (-0.01, -1, 1)

# tested_size = np.arange(1,7)*7

# method = [["geometric", 0.999], ["geometric", 0.99], ["geometric",0.9], ["arithmetic", 0.001], ["arithmetic", 0.01], ["arithmetic", 0.1]]

# err_q = []
# ri_q = []

# for i in range(len(method)):
    
#     name = "{}x{}".format(n,n)
#     randommap = generate_random_map(size = n, p= prob)  # creates dictionary
#     term, goal, name = return_term_goal_name(randommap)
#     env_kwargs = {
#         'desc': randommap,
#         'slip_rate': sliprate,
#         'rewards': rewards
#     }
#     env = FrozenLakeEnv(**env_kwargs)
#     pi_env = env.unwrapped
#     P = pymdp_PR(pi_env)[0]
#     R = pymdp_PR(pi_env)[1]
#     q = QLearning(P, R, gamma = gamma, n_iter = 10000, epsilon_min =0.00001, epsilon_decay_method= method[i][0], epsilon_decay= method[i][1], )
#     q.run()
#     err_q.append(q.err)
#     ri_q.append(q.rt_per_iter )


# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("Decay strategy: Q Learning Error Convergence 25x25")
# plt.ylabel("Error")
# plt.xlabel("Iteration")
# plt.grid(linewidth = 0.2)  
# for i in range(len(method)):  
#     plt.plot(np.arange(1, len(err_q[i]) +1 ) , err_q[i], label=method[i][0] + ", rate = {}".format(method[i][1]))
# plt.xlim(-10,2000)
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Decay strategy: Cumulative Runtime")
# plt.ylabel("Runtime in seconds")
# plt.xlabel("completion rate of iterations")
# plt.grid(linewidth = 0.2)    
# for i in range(len(method)):  
#     # print("The number of iteration: gridsize: {}x{}, pi: {}".format(e,e,len((ri_pi[i]))))
#     # print("Total runtime: gridsize: {}x{}, pi: {}".format(e,e,np.sum(ri_pi[i])))    
#     # print("The number of iteration: gridsize: {}x{}, vi: {}".format(e,e,len((ri_vi[i]))))
#     # print("Total runtime: gridsize: {}x{}, vi: {}".format(e,e,np.sum(ri_vi[i])))    
#     print("The number of iteration for " + method[i][0] + ", r = {}".format(method[i][1]) + ": {}".format(len((ri_q[i]))))
#     print("Total runtime for " + method[i][0] + ", r = {}".format(method[i][1]) +  ": {}".format(np.sum(ri_q[i])))
#     print("Mean runtime per iteration for " + method[i][0] + ", r = {}".format(method[i][1]) +  ": {}".format(np.mean(ri_q[i])))
#     # plt.plot(np.linspace(0, 1, num = len(ri_pi[i])), np.cumsum(ri_pi[i]), label="gridsize: {}x{}, policy iter.".format(e,e), linewidth =1)
#     # plt.plot(np.linspace(0, 1, num = len(ri_vi[i])), np.cumsum(ri_vi[i]), label="gridsize: {}x{}, value iter.".format(e,e), linewidth =1)
#     plt.plot(np.linspace(0, 1, num = len(ri_q[i])), np.cumsum(ri_q[i]), label=method[i][0] + ", rate = {}".format(method[i][1]))
# # plt.ylim(-1,5)
# # plt.xscale("log")
# plt.legend() 
# plt.savefig("Epsilon decay Q convergence and runtime")


# Alpha Decau



n = 25
sliprate = 0.2
prob = 0.8
gamma = 0.9
theta = 0.0001
rewards = (-0.01, -1, 1)

tested_size = np.arange(1,7)*7

method = [["geometric", 0.9999], ["geometric", 0.999], ["geometric",0.99],["arithmetic", 0.0000005], ["arithmetic", 0.000001], ["arithmetic", 0.0000015], ]#["arithmetic", 0.000002],] #["arithmetic", 0]]

err_q = []
ri_q = []

for i in range(len(method)):
    
    name = "{}x{}".format(n,n)
    randommap = generate_random_map(size = n, p= prob)  # creates dictionary
    term, goal, name = return_term_goal_name(randommap)
    env_kwargs = {
        'desc': randommap,
        'slip_rate': sliprate,
        'rewards': rewards
    }
    env = FrozenLakeEnv(**env_kwargs)
    pi_env = env.unwrapped
    P = pymdp_PR(pi_env)[0]
    R = pymdp_PR(pi_env)[1]
    q = QLearning(P, R, gamma = gamma, n_iter = 50000 , alpha = 0.1 , alpha_decay_method = method[i][0], alpha_decay = method[i][1], )
    q.run()
    err_q.append(q.err)
    ri_q.append(q.rt_per_iter )


plt.figure(0, figsize = (12,5))
plt.subplot(1,2,1)
plt.title("Alpha Decay: Q Learning Error Convergence 10x10")
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.grid(linewidth = 0.2)  
for i in range(len(method)):  
    plt.plot(np.arange(1, len(err_q[i]) +1 ) , err_q[i], label=method[i][0] + ", rate = {}".format(method[i][1]), linewidth = 0.3)
plt.xlim(-10,50000)
plt.legend()

plt.subplot(1,2,2)
plt.title("Alpha Decay: Cumulative Runtime 10x10")
plt.ylabel("Runtime in seconds")
plt.xlabel("completion rate of iterations")
plt.grid(linewidth = 0.2)    
for i in range(len(method)):  
    # print("The number of iteration: gridsize: {}x{}, pi: {}".format(e,e,len((ri_pi[i]))))
    # print("Total runtime: gridsize: {}x{}, pi: {}".format(e,e,np.sum(ri_pi[i])))    
    # print("The number of iteration: gridsize: {}x{}, vi: {}".format(e,e,len((ri_vi[i]))))
    # print("Total runtime: gridsize: {}x{}, vi: {}".format(e,e,np.sum(ri_vi[i])))    
    print("The number of iteration for " + method[i][0] + ", r = {}".format(method[i][1]) + ": {}".format(len((ri_q[i]))))
    print("Total runtime for " + method[i][0] + ", r = {}".format(method[i][1]) +  ": {}".format(np.sum(ri_q[i])))
    print("Mean runtime per iteration for " + method[i][0] + ", r = {}".format(method[i][1]) +  ": {}".format(np.mean(ri_q[i])))
    # plt.plot(np.linspace(0, 1, num = len(ri_pi[i])), np.cumsum(ri_pi[i]), label="gridsize: {}x{}, policy iter.".format(e,e), linewidth =1)
    # plt.plot(np.linspace(0, 1, num = len(ri_vi[i])), np.cumsum(ri_vi[i]), label="gridsize: {}x{}, value iter.".format(e,e), linewidth =1)
    plt.plot(np.linspace(0, 1, num = len(ri_q[i])), np.cumsum(ri_q[i]), label=method[i][0] + ", rate = {}".format(method[i][1]))
# plt.ylim(-1,5)
# plt.xscale("log")
plt.legend() 
plt.savefig("Alpha Decay Q convergence and runtime")






# ALGO = 'vi'

# n = 6
# prob = 0.8
# gamma = 0.9
# theta = 0.0001
# rewards = (-0.01, -1, 1)
# name = "{}x{}".format(n,n)
# randommap = generate_random_map(size = n, p= prob)  # creates dictionary
# term, goal, name = return_term_goal_name(randommap)
# # print(name)
# env_kwargs = {
#     'desc': randommap,
#     'slip_rate': 0.2,
#     'rewards': rewards
# }

# threshold = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# gammas = np.linspace(.01, 0.99, 30)
# n_trials = 10

# n_iters = {k: [] for k in threshold}
# runtimes = {k: [] for k in threshold}

# for theta in threshold:
#     for gamma in gammas:
#         temp_n_iters = []
#         temp_runtimes = []
#         for t in range(n_trials):
#             env = FrozenLakeEnv(**env_kwargs)
#             env = env.unwrapped
#             P = pymdp_PR(env)[0]
#             R = pymdp_PR(env)[1]
#             vi = ValueIteration(P, R, gamma = gamma,) #n_iter = 50000 , alpha = 0.1 , alpha_decay_method = method[i][0], alpha_decay = method[i][1], )
#             vi.run()
#             n_iter = vi.iter
#             runtime = vi.time
#             temp_n_iters.append(n_iter)
#             temp_runtimes.append(runtime)
#         n_iters[theta].append(np.mean(temp_n_iters))
#         runtimes[theta].append(np.mean(temp_runtimes))

# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# for item, iterlist in n_iters.items():
#     plt.plot(gammas, iterlist, label=('theta = {}'.format(item)))
# plt.title('Discount Factor: VI Iterations (5x5)')
# plt.legend(loc='upper left')
# plt.xlabel('Gamma')
# plt.ylabel('Iterations')
# plt.grid(linewidth = 0.2)  
# plt.subplot(1,2,2)
# for item, rt in runtimes.items():
#     plt.plot(gammas, [t for t in rt], label=('theta = {}'.format(item)))
# plt.title('Discount Factor: VI Time (5x5)')
# plt.legend(loc='upper left')
# plt.xlabel('Gamma')
# plt.ylabel('Runtime')
# plt.grid(linewidth = 0.2)  
# plt.savefig('gamma experiment')
# plt.show()


# gammas = np.linspace(0.001, 0.999, 20)
# n = 12
# prob = 0.8
# gamma = 0.9
# theta = 0.0001
# rewards = (-0.01, -1, 1)
# name = "{}x{}".format(n,n)
# randommap = generate_random_map(size = n, p= prob)  # creates dictionary
# term, goal, name = return_term_goal_name(randommap)
# # print(name)
# env_kwargs = {
#     'desc': randommap,
#     'slip_rate': 0.2,
#     'rewards': rewards
# }
# pi_rt = []
# vi_rt = []
# # q_rt = []
# pi_iter = []
# vi_iter = []

    
# for gamma in gammas:
  
#     env = FrozenLakeEnv(**env_kwargs)
#     env = env.unwrapped
#     P = pymdp_PR(env)[0]
#     R = pymdp_PR(env)[1]
#     vi = ValueIteration(P, R, gamma = gamma,) #n_iter = 50000 , alpha = 0.1 , alpha_decay_method = method[i][0], alpha_decay = method[i][1], )
#     vi.run()    
#     pi = PolicyIterationModified(P, R, gamma = gamma,)
#     pi.run()
#     pi_rt.append(pi.time)
#     vi_rt.append(vi.time)
#     pi_iter.append(pi.iter)
#     vi_iter.append(vi.iter)
    
    
# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("Gamma vs # of iteration (Frozen) gridsize 12x12")
# plt.ylabel("iteration")
# plt.xlabel("gamma")
# plt.grid(linewidth = 0.2)  
# plt.plot(gammas, pi_iter, label="policy iter, s = 12x12")
# plt.plot(gammas, vi_iter, label="value iters, s = 12x12")
# # plt.plot(gammas, pi_iter2, label="policy iter, s = 1000")
# # plt.plot(gammas, vi_iter2, label="value iters, s = 1000")
# # plt.plot(gammas, q_iter, label="q learning")    
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Gamma vs runtime (Frozen) gridsize 6x6")
# plt.ylabel("Runtime in seconds")
# plt.xlabel("gamma")
# plt.grid(linewidth = 0.2)    
# plt.plot(gammas, np.cumsum(pi_rt), label="policy iter, s = 12x12")
# plt.plot(gammas, np.cumsum(vi_rt), label="value iter, s = 12x12")
# # plt.plot(gammas, np.cumsum(pi_rt2), label="policy iter, s = 1000")
# # plt.plot(gammas, np.cumsum(vi_rt2), label="value iter, s = 1000")
# # plt.plot(gammas, q_rt, label="q learning")    
# plt.legend() 
# plt.savefig("Frozen: iteration convergence and runtime")
