# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 08:00:31 2020

@author: nicholas.park
"""

from hiive.mdptoolbox.mdp import *
from hiive.mdptoolbox.example import *
from hiive.visualization import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpers import visualize_policy, visualize_value, visualize_env
from constants import design_random_map
from policy_iteration import policy_iteration
from value_iteration import value_iteration
from q_learning import q_learning
import plotting as plotting
plt.style.use("default")
plt.close()

# reward = [4,2]

# runtimes = []
# iters = []
# error = []

# tested_maxage = np.arange(1,8)*3
# tested_burnprob = np.linspace(0.001,0.999,num=30)

# for n in tested_maxage:
    
#     runtimes_burnprob =  []
#     iters_burnprob =  []
#     errors_burnprob = []
    
#     for burnprob in tested_burnprob:
        
#         prob = 0.8
#         gamma = 0.9
#         theta = 0.0001
#         rewards = (-0.01, -1, 1)
#         P, R = forest(S=n, r1=reward[0], r2=reward[1], p=burnprob, is_sparse=False)
#         pi = PolicyIterationModified(P, R, gamma, epsilon = theta, )
#         pi.run()
#         # print(pi.iter)
#         # print(pi.time)
#         n_iter = pi.iter
#         runtime = pi.time
#         errors_burnprob.append(pi.err)
#         iters_burnprob.append(n_iter)
#         runtimes_burnprob.append(runtime)

#     runtimes.append(runtimes_burnprob)
#     iters.append(iters_burnprob)
#     error.append(errors_burnprob)


# # print(runtimes_sliprate)

# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("Policy Iter.: # of iters. vs burn prob.")
# plt.ylabel("Number of iteration")
# plt.xlabel("Burn Probability")
# plt.grid(linewidth = 0.2)    
# for i, iters_burnprob in enumerate(iters):
#     plt.plot(tested_burnprob, iters_burnprob, label="# of states = {}".format(tested_maxage[i]))
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Policy Iter.: cum. runtime vs burn prob.")
# plt.ylabel("Runtime")
# plt.xlabel("Burn Probability")
# plt.grid(linewidth = 0.2)    
# for i, runtimes_burnprob in enumerate(runtimes):
#     plt.plot(tested_burnprob,np.cumsum(runtimes_burnprob), label="# of states = {}".format(tested_maxage[i]))
# plt.legend()
# plt.savefig("burnprob experiment (PI)")


# runtimes = []
# iters = []


# for n in tested_maxage:
    
#     runtimes_burnprob =  []
#     iters_burnprob =  []
#     errors_burnprob = []
    
#     for burnprob in tested_burnprob:
        
#         gamma = 0.9
#         theta = 0.0001
#         P, R = forest(S=n, r1=reward[0], r2=reward[1], p=burnprob, is_sparse=False)
#         pi = ValueIteration(P, R, gamma, epsilon = theta, )
#         pi.run()
#         # print(pi.iter)
#         # print(pi.time)
#         n_iter = pi.iter
#         runtime = pi.time
#         errors_burnprob.append(pi.err)
#         iters_burnprob.append(n_iter)
#         runtimes_burnprob.append(runtime)

#     runtimes.append(runtimes_burnprob)
#     iters.append(iters_burnprob)
#     error.append(errors_burnprob)

# # print(runtimes_sliprate)

# plt.figure(2, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("Value Iter.: # of iters. vs burn prob.")
# plt.ylabel("Number of iteration")
# plt.xlabel("Burn Probability")
# plt.grid(linewidth = 0.2)    
# for i, iters_burnprob in enumerate(iters):
#     plt.plot(tested_burnprob, iters_burnprob, label="# of states = {}".format(tested_maxage[i]))
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Value Iter.: cum. runtime vs burn prob.")
# plt.ylabel("Runtime")
# plt.xlabel("Burn Probability")
# plt.grid(linewidth = 0.2)    
# for i, runtimes_burnprob in enumerate(runtimes):
#     plt.plot(tested_burnprob,np.cumsum(runtimes_burnprob), label="# of states = {}".format(tested_maxage[i]))
# plt.legend()
# plt.savefig("burnprob experiment (VI)")


# rewards = [[4,2], [2,4], [100,1], [1,100], [3,3], [50, 50]]
# n = 28
# burnprob = 0.1
# for reward in rewards:
#     gamma = 0.9
#     theta = 0.0001
#     P, R = forest(S=n, r1=reward[0],p= burnprob, r2=reward[1], is_sparse=False)
#     pi = PolicyIterationModified(P, R, gamma, epsilon = theta, )
#     pi.run()
#     # print(np.around(list(pi.policy),2))
#     print(pi.policy)
#     print("--------")



# matchrate experiment


# tested_maxage = [12, 36, 108, 324, 972]
# tested_burnprob = np.linspace(0.001,0.999,num=20)
# tested_seeds = np.arange(1,30)

# gamma = 0.9
# reward = [4,2]

# match_rate_per_size_policy = []
# match_rate_per_size_v = []
# atol = 0.1

# for n in tested_maxage:

#     total_policy_match= []
#     total_value_match= []

#     for seed in tested_seeds:

#         policy_match = []
#         value_match = []
        
#         for burnprob in tested_burnprob:
            
#             # theta = 0.0001
#             P, R = forest(S=n, r1=reward[0], r2=reward[1], p=burnprob, is_sparse=False)
#             vi = ValueIteration(P, R, gamma =gamma,)
#             pi = PolicyIterationModified(P, R, gamma = gamma, )
#             pi.run()
#             vi.run()
#             policy_match.append( int(np.all(np.isclose(pi.policy, vi.policy, atol=atol)) ))
#             value_match.append(int(np.all(np.isclose(pi.V, vi.V, atol=atol))  ))
#         # print(policy_match)
            
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
# plt.title("PI and VI: Policy Match Rate (atol = 0.1)")
# plt.ylabel("Match Rate")
# plt.xlabel("Burn Probability")
# plt.grid(linewidth = 0.2)    
# for i, e in enumerate(match_rate_per_size_policy):
#     plt.plot(tested_burnprob, e, label="# of states = {}".format(tested_maxage[i]))
# # plt.ylim(0, 1.05)
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("PI and VI: Value Match Rate (atol = 0.1)")
# plt.ylabel("Match Rate")
# plt.xlabel("Burn Probability")
# plt.grid(linewidth = 0.2)    
# for i, e in enumerate(match_rate_per_size_v):
#     plt.plot(tested_burnprob, e, label="# of states = {}".format(tested_maxage[i]))
# # plt.ylim(0, 1.05)
# plt.legend()
# plt.savefig("PI_VI_matchrate_experiment_forest")



# burnprob = 0.1
# gamma = 0.9
# theta = 0.0001
# reward = [4,2]

# tested_size = [30,270,720] 

# err_vi = []
# err_pi = []
# err_q = []
# ri_vi = []
# ri_pi = []
# ri_q = []

# for size  in tested_size:
  
#     P, R = forest(S=size, r1=reward[0], r2=reward[1], p=burnprob, is_sparse=False)
#     pi = PolicyIteration(P, R, gamma = gamma,)
#     vi = ValueIteration(P, R, gamma =gamma,epsilon=0.01)
#     q = QLearning(P, R, gamma = gamma, n_iter = 50000)
#     pi.run()
#     vi.run()
#     # print(vi.err)
#     q.run()
#     err_pi.append(pi.err)
#     err_vi.append(vi.err)
#     err_q.append(q.err)
#     ri_vi.append(vi.rt_per_iter)
#     ri_pi.append(pi.rt_per_iter)
#     ri_q.append(q.rt_per_iter )


# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title(" Error Convergence")
# plt.ylabel("Error")
# plt.xlabel("Iteration")
# plt.grid(linewidth = 0.2)  
# for i,e in enumerate(tested_size):  
#     # plt.plot(np.linspace(0, 1, num = len(err_pi[i])) , err_pi[i], label="PI: # of states = {}".format(tested_size[i]), linewidth = 0.3)
#     # plt.plot(np.linspace(0, 1, num = len(err_vi[i])), err_vi[i], label="VI: # of states = {}".format(tested_size[i]), linewidth = 0.3)
#     # plt.plot(np.linspace(0, 1, num = len(err_q[i])), err_q[i], label="# of states = {}".format(tested_size[i]), linewidth = 0.3)
#     plt.plot(np.arange(1, len(err_pi[i]) +1 ) , err_pi[i], label="PI: # of states = {}".format(tested_size[i]), linewidth = 1)
#     plt.plot(np.arange(1, len(err_vi[i]) +1 ) , err_vi[i], label="VI: # of states = {}".format(tested_size[i]), linewidth = 1)
#     plt.plot(np.arange(1, len(err_q[i]) +1 ) , err_q[i], label="Q: # of states = {}".format(tested_size[i]), linewidth = 0.55)
# plt.ylim(-0.0005,0.5)
# plt.xlim(0, 50)
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Cumulative Runtime (Forest)")
# plt.ylabel("Runtime in seconds")
# plt.xlabel("Iteration completion rate (in %)")
# plt.grid(linewidth = 0.2)    
# for i,e in enumerate(tested_size):  
#     # plt.plot(np.arange(1, len(ri_pi[i]) +1 ) , ri_pi[i], label="PI: # of states = {}".format(tested_size[i]), linewidth = 1)
#     # plt.plot(np.arange(1, len(ri_vi[i]) +1 ) , ri_vi[i], label="VI: # of states = {}".format(tested_size[i]), linewidth = 1)
#     # plt.plot(np.arange(1, len(ri_q[i]) +1 ) , ri_q[i], label="Q: # of states = {}".format(tested_size[i]), linewidth = 1)
#     plt.plot(np.linspace(0,1,len(ri_pi[i])) , ri_pi[i], label="PI: # of states = {}".format(tested_size[i]), linewidth = 1)
#     plt.plot(np.linspace(0,1,len(ri_vi[i]))  , ri_vi[i], label="VI: # of states = {}".format(tested_size[i]), linewidth = 1)
#     plt.plot(np.linspace(0,1,len(ri_q[i]))  , ri_q[i], label="Q: # of states = {}".format(tested_size[i]), linewidth = 0.3)
#     # print("The number of iteration: state size: {}, pi: {}".format(e,len((ri_pi[i]))))
#     # print("Average runtime per iteration: state size: {}, pi: {}".format(e,np.mean(ri_pi[i])))    
#     # print("The number of iteration: state size: {}, vi: {}".format(e,len((ri_vi[i]))))
#     # print("Average runtime per iteration: state size: {}, vi: {}".format(e,np.mean(ri_vi[i])))
#     print("The number of iteration: state size: {}, Q: {}".format(e,len((ri_q[i]))))
#     print("Average runtime per iteration: state size: {}, Q: {}".format(e,np.mean(ri_q[i])))
#     # plt.plot(np.linspace(0, 1, num = len(ri_q[i])), ri_q[i], label="state size: {}, q learning".format(e), linewidth = 0.13)
# plt.ylim(0,0.002)
# # plt.xscale("log")
# plt.legend() 

# plt.savefig("Forest: Q VI PI iteration convergence and runtime")



# gamma exploring

# burnprob = 0.1
# gammas = np.linspace(0.001, 0.999, 25)
# theta = 0.0001
# reward = [4,2]

# pi_rt = []
# vi_rt = []
# q_rt = []
# pi_iter = []
# vi_iter = []
# # q_iter = []
# pi_rt2 = []
# vi_rt2 = []
# # q_rt = []
# pi_iter2 = []
# vi_iter2 = []
# # q_iter = []
    
# for gamma in gammas:
  
#     P, R = forest(S=100, r1=reward[0], r2=reward[1], p=burnprob, is_sparse=False)
#     pi = PolicyIterationModified(P, R, gamma = gamma,)
#     vi = ValueIteration(P, R, gamma =gamma,)
#     # q = QLearning(P, R, gamma = gamma, n_iter = 50000)
#     pi.run()
#     vi.run()
#     # q.run()
#     pi_rt.append(pi.time)
#     vi_rt.append(vi.time)
#     # q_rt.append(q.time)
#     pi_iter.append(pi.iter)
#     vi_iter.append(vi.iter)
#     # q_iter.append(q.iter)

#     P, R = forest(S=1000, r1=reward[0], r2=reward[1], p=burnprob, is_sparse=False)
#     pi = PolicyIterationModified(P, R, gamma = gamma,)
#     vi = ValueIteration(P, R, gamma =gamma,)
#     # q = QLearning(P, R, gamma = gamma, n_iter = 50000)
#     pi.run()
#     vi.run()
#     # q.run()
#     pi_rt2.append(pi.time)
#     vi_rt2.append(vi.time)
#     # q_rt.append(q.time)
#     pi_iter2.append(pi.iter)
#     vi_iter2.append(vi.iter)
#     # q_iter.append(q.iter)

    
    
# plt.figure(0, figsize = (12,5))
# plt.subplot(1,2,1)
# plt.title("Gamma vs # of iteration (Forest) states = 1000")
# plt.ylabel("iteration")
# plt.xlabel("gamma")
# plt.grid(linewidth = 0.2)  
# # plt.plot(gammas, pi_iter, label="policy iter, s = 100")
# # plt.plot(gammas, vi_iter, label="value iters, s = 100")
# plt.plot(gammas, pi_iter2, label="policy iter, s = 1000")
# plt.plot(gammas, vi_iter2, label="value iters, s = 1000")
# # plt.plot(gammas, q_iter, label="q learning")    
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Gamma vs runtime (Forest) states = 1000")
# plt.ylabel("Runtime in seconds")
# plt.xlabel("gamma")
# plt.grid(linewidth = 0.2)    
# # plt.plot(gammas, np.cumsum(pi_rt), label="policy iter, s = 100")
# # plt.plot(gammas, np.cumsum(vi_rt), label="value iter, s = 100")
# plt.plot(gammas, np.cumsum(pi_rt2), label="policy iter, s = 1000")
# plt.plot(gammas, np.cumsum(vi_rt2), label="value iter, s = 1000")
# # plt.plot(gammas, q_rt, label="q learning")    
# plt.legend() 
# plt.savefig("Forest: VI iteration convergence and runtime")





# n = 25
burnprob = 0.1
prob = 0.8
gamma = 0.9
theta = 0.0001
reward = [4,2]

size = 200
tested_size = np.arange(1,7)*7

method = [["geometric", 0.999], ["geometric", 0.99], ["geometric",0.9], ["arithmetic", 0.001], ["arithmetic", 0.01], ["arithmetic", 0.1]]

err_q = []
ri_q = []

for i in range(len(method)):
    
    P, R = forest(S=size, r1=reward[0], r2=reward[1], p=burnprob, is_sparse=False)
    q = QLearning(P, R, gamma = gamma, n_iter = 50000, alpha_min =0.00000001, alpha_decay_method= method[i][0], alpha_decay= method[i][1], )
    q.run()
    err_q.append(q.err)
    ri_q.append(q.rt_per_iter )


plt.figure(0, figsize = (12,5))
plt.subplot(1,2,1)
plt.title("Alpha Decay: Q Learning Error Convergence, s = 200")
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.grid(linewidth = 0.2)  
for i in range(len(method)):  
    plt.plot(np.arange(1, len(err_q[i]) +1 ) , err_q[i], label=method[i][0] + ", rate = {}".format(method[i][1]), linewidth = 0.4)
plt.xlim(-10,50000)
# plt.ylim(0.00001,0.03)
plt.legend()

plt.subplot(1,2,2)
plt.title("Alpha Decay: Cumulative Runtime, s = 200")
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
plt.savefig("Forest Alpha decay Q convergence and runtime")








