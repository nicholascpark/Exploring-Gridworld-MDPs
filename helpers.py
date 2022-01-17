'''
Adopted from the reinforcement learning resource repository of Denny Britz @
https://github.com/dennybritz/reinforcement-learning
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

    

def visualize_env(env, term, goal, name, title = None):
    
    TERM_STATE_MAP = term
    GOAL_STATE_MAP = goal
    env_name = name
    shape = env.desc.shape
    M = shape[0]
    N = shape[1]
    arr = np.zeros(shape)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                arr[i, j] = 0.25
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                arr[i, j] = 1.0
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap='autumn_r')
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)

    for i in range(M):
        for j in range(N):
            if (i, j) == (0, 0):
                ax.text(j, i, 'S', ha='center', va='center', color='k', size=18)
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k', size=18)
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k', size=18)
            else:
                pass
    fig.tight_layout()
    if title:
        ax.set_title(title)
    plt.show()


def visualize_policy(pi, term, goal, name, shape, title=None):
    
    TERM_STATE_MAP = term
    GOAL_STATE_MAP = goal
    env_name = name
    M = shape[0]
    N = shape[1]
    actions = np.argmax(pi, axis=1).reshape(shape)
    mapping = {
        0: '<',
        1: 'v',
        2: '>',
        3: '^'
    }
    arr = np.zeros(shape)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                arr[i, j] = 0.25
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                arr[i, j] = 1.0
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap='autumn_r')
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)

    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k', size=18)
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k', size=18)
            else:
                ax.text(j, i, mapping[actions[i, j]], ha='center', va='center', color='k', size=18)
    #fig.tight_layout()
    if title:
        ax.set_title(title)
    plt.show()

def render_policy(pi, term, goal, name, shape):

    TERM_STATE_MAP = term
    GOAL_STATE_MAP = goal
    env_name = name
    actions = np.argmax(pi, axis=1)
    for index in TERM_STATE_MAP[env_name]:
        actions[index] = 999
    for index in GOAL_STATE_MAP[env_name]:
        actions[index] = 1000

    pi = np.reshape(actions, shape)

    mapping = {
        0: ' < ',
        1: ' v ',
        2: ' > ',
        3: ' ^ ',
        999: ' . ',
        1000: ' $ '
    }
    mapper = np.vectorize(lambda k: mapping[k])
    np.apply_along_axis(lambda row: print(' '.join(row)), axis=1, arr=mapper(pi))


def visualize_value(V, term, goal, name, shape, title=None):

    TERM_STATE_MAP = term
    GOAL_STATE_MAP = goal
    env_name = name
    M = shape[0]
    N = shape[1]
    fig, ax = plt.subplots()

    #V[TERM_STATE_MAP[env_name]] = 
    arr = V.reshape(shape)
    arr[-1,-1] = 1
    im = ax.imshow(arr, cmap='autumn_r')
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k')
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k')
            else:
                ax.text(j, i, '%.2f' % (arr[i, j]), ha='center', va='center', color='k')
    fig.tight_layout()
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('State-value estimate', rotation=-90, va="bottom")
    if title:
        ax.set_title(title)
    plt.show()

def better_desc(desc):
    mapping = {
        b'S': b' S ',
        b'F': b' * ',
        b'H': b' O ',
        b'G': b' $ '
    }
    mapper = np.vectorize(lambda k: mapping[k])
    return mapper(desc)