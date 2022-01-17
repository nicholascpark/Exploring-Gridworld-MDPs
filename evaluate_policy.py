# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:49:09 2020

@author: nicholas.park
"""
import numpy as np

def evaluate_policy(self, policy, discount_factor=1.0, max_steps=None, theta=0.00001):

    env = self.get_environment()

    V = np.zeros(env.nS) 
    steps = 0
    while max_steps is None or steps < max_steps:
        delta = 0
        for i_s in range(env.nS):
            v = 0
            for i_a, action_prob in enumerate(policy[i_s]):
                try:
                    for prob, i_next_state, reward, done in env.P[i_s][i_a]:
                        v += action_prob * prob * (reward + discount_factor * V[i_next_state])
                except KeyError:
                    qualfied_state = env.index_to_state[i_s]
                    qualified_action = env.index_to_action[i_a]
                    for prob, qualfied_next_state, reward, done in env.P[qualfied_state][qualified_action]:
                        i_next_state = env.state_to_index[qualfied_next_state]
                        v += action_prob * prob * (reward + discount_factor * V[i_next_state])
            delta = max(delta, np.abs(v - V[i_s]))
            V[i_s] = v
        steps += 1
        if delta < theta:
            break
    return np.array(V)