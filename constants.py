import numpy as np

def design_random_map(size = 4):
    TERM_STATE_MAP ={}
    GOAL_STATE_MAP ={}
    
    # for i in range(4, 50):
    i = size
    maxpossible = i**2-1
    num_to_draw_from = np.arange(2, maxpossible)
    num_to_avoid = [2, i, i**2-2, i**2-1-i]
    num_to_draw_from = [j for j in num_to_draw_from if j not in num_to_avoid]
    num_obstacles = int((i**2)*0.25)
    
    obstacles = np.sort(np.random.choice(num_to_draw_from, num_obstacles))
    goal = []
    goal.append(i**2-1)
    
    envname = "{}x{}".format(i,i)
    
    TERM_STATE_MAP[envname] = obstacles
    GOAL_STATE_MAP[envname] = goal
        
    return TERM_STATE_MAP, GOAL_STATE_MAP, envname
