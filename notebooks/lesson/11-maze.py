import matplotlib.pyplot as plt
from scipy.special import softmax
import sys
import numpy as np
import argparse

import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

actions = [np.array(x) for x in [(1,0), (-1,0), (0,1), (0,-1)]]

SLIP_PROB=0.6

ASP_GREEDY = "greedy"
ASP_EPS_GREEDY = "egreedy"
ASP_SOFTMAX = "softmax"

class Maze:

    def __init__ (self, size=(5,5), blocked_cells=None, slippery_cells=None):
        assert(len(size)==2)
        self.size = size
        self.cells = np.zeros(size)

        if slippery_cells is not None:
            for i,j in slippery_cells:
                self.cells[i,j] = 2

        if blocked_cells is not None:
            for i,j in blocked_cells:
                self.cells[i,j] = 1


    def is_blocked (self, cell):
        return self.cells[tuple(cell)] == 1

    def is_slippery (self, cell):
        return self.cells[tuple(cell)] == 2

    def is_valid (self, cell):
        return np.all(cell < self.size) and np.all(cell >= 0) and not self.is_blocked(cell)

    def render(self, positions, final_pos, fps=3, title=None):
        fig = plt.figure( figsize=(8,8) )
        if title is not None:
            fig.suptitle(title)

        cmap = ListedColormap(["white", "black", "#dddddd", "red", "yellow"])
        im = plt.imshow(self.cells, interpolation='none', aspect='auto', vmin=0, vmax=4, cmap=cmap)

        def animate_func(i):
            cells = self.cells.copy()
            cells[tuple(positions[i])] = 3
            cells[tuple(final_pos)] = 4

            im.set_array(cells)
            return [im]

        anim = animation.FuncAnimation(
                                    fig, 
                                    animate_func, 
                                    frames = len(positions),
                                    interval = 1000 / fps, # in ms
                                    )

        #anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

        plt.show()


class RandomAgent():

    def __init__ (self, maze):
        self.maze = maze

    def next_action (self, pos):
        return actions[np.random.randint(0, len(actions))]

    def update (self, reward, pos, action, new_pos):
        pass

class MDPBasedAgent():

    def __init__ (self, maze):
        self.maze = maze
        self.Qtable = {}

        self.value_iteration()

    def getQ (self, state, action):
        return self.Qtable.get((tuple(state), tuple(action)), 0.0)

    def setQ (self, state, action, value):
        self.Qtable[(tuple(state), tuple(action))] = value

    def value_iteration (self):
        delta = float("inf")
        iters = 0
        while delta > 0.0001:
            delta = self._value_iteration()
            iters += 1
        print(f"VI iterations: {iters}")

    def _value_iteration (self):
        max_delta = 0.0
        final_pos = np.array([0, self.maze.size[1]-1])
        for i, j in np.ndindex(self.maze.cells.shape):
            pos = np.array((i,j))
            for a in actions:
                q = 0
                # immediate reward
                new_pos = pos + a
                if not self.maze.is_valid(new_pos):
                    q += -1000
                else:
                    if not np.array_equal(new_pos, final_pos):
                        q += -1 + self.getQ(new_pos, self.greedy_action(new_pos)) 
                        if self.maze.is_slippery(new_pos):
                            # we have two possible next states to account for
                            q *= (1-SLIP_PROB)

                            new_pos = new_pos + a
                            if not self.maze.is_valid(new_pos):
                                q += -1000*SLIP_PROB
                            elif not np.array_equal(new_pos, final_pos):
                                q += SLIP_PROB*(-1 + self.getQ(new_pos, self.greedy_action(new_pos)))

                delta = abs(q - self.getQ(pos, a))
                max_delta = max(delta, max_delta)
                self.setQ(pos, a, q)
        return max_delta

    def next_action (self, pos):
        return self.greedy_action(pos)

    def greedy_action (self, pos):
        best_action = None
        best_q = None

        for a in actions:
            q = self.getQ(pos,a)
            if best_action is None or q > best_q:
                best_q = q
                best_action = a
        return best_action


    def update (self, reward, state, action, new_state):
        pass

class QlearningAgent():

    def __init__ (self, maze, alpha=0.1, epsilon=0.1, action_selection_policy="softmax"):
        self.maze = maze
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_selection_policy = action_selection_policy
        self.Qtable = {}

        self.last_state = None
        self.last_action = None

    def getQ (self, state, action):
        return self.Qtable.get((tuple(state), tuple(action)), 0.0)

    def setQ (self, state, action, value):
        self.Qtable[(tuple(state), tuple(action))] = value

    def next_action (self, pos):
        if self.action_selection_policy == ASP_GREEDY:
            a  = self._greedy_action(pos)
        elif self.action_selection_policy == ASP_EPS_GREEDY:
            if np.random.random() <= self.epsilon:
                a = self._random_action(pos)
            else:
                a  = self._greedy_action(pos)
        elif self.action_selection_policy == ASP_SOFTMAX:
            a = self._softmax_action(pos)

        self.last_action = a
        self.last_state = pos

        return a

    def _softmax_action (self, state):
        p = softmax([self.getQ(state,a) for a in actions])
        return actions[np.random.choice(np.arange(len(actions)), p=p)]

    def _greedy_action (self, state):
        best_action = None
        best_q = None

        for a in actions:
            q = self.getQ(state,a)
            if best_action is None or q > best_q:
                best_q = q
                best_action = a
        return best_action

    def _random_action (self, state):
        return actions[np.random.randint(0, len(actions))]

    def update (self, reward, state, action, new_state):
        old_q = self.getQ(state,action)
        next_a = self._greedy_action(new_state)
        new_q = (1-self.alpha)*old_q + self.alpha*(reward + self.getQ(new_state,next_a))
        self.setQ(state,action,new_q)

class SARSAAgent(QlearningAgent):

    def __init__ (self, maze, alpha=0.1, epsilon=0.1, action_selection_policy="softmax"):
        super().__init__(maze, alpha, epsilon, action_selection_policy)
        self.next_chosen_action = None
        self.next_state = None

    def next_action (self, pos):
        if self.next_chosen_action is None or np.any(pos != self.next_state):
            return super().next_action(pos)
        else:
            return self.next_chosen_action
    
    def update (self, reward, state, action, new_state):
        old_q = self.getQ(state,action)
        self.next_chosen_action = super().next_action(new_state)
        self.new_state = new_state
        new_q = (1-self.alpha)*old_q + self.alpha*(reward + self.getQ(new_state,self.next_chosen_action))
        self.setQ(state,action,new_q)


def run_episode (maze, agent, final_pos, random_initial_pos=False, 
        reward_model=0, max_moves=1000):

    if reward_model == 0:
        MOVE_REWARD = -1
        CRASH_REWARD = -1000
        GOAL_REWARD = 0
    elif reward_model == 1:
        MOVE_REWARD = 0
        CRASH_REWARD = 0
        GOAL_REWARD = 1
    elif reward_model == 2:
        MOVE_REWARD = 0
        CRASH_REWARD = -1
        GOAL_REWARD = 1

    if random_initial_pos:
        pos = np.array([np.random.randint(0,maze.size[0]),0])
    else:
        pos = np.array([0,0])
    positions = [pos]
    total_reward = 0
    moves = 0

    while not np.array_equal(pos, final_pos) and moves < max_moves:
        a = agent.next_action(pos)

        new_pos = pos + a
        moves += 1

        # slip with some probability
        if maze.is_valid(new_pos) and maze.is_slippery(new_pos) and np.random.random() <= SLIP_PROB:
            new_pos = new_pos + a

        if not maze.is_valid(new_pos):
            reward = CRASH_REWARD
            total_reward += reward
            agent.update(reward, pos, a, pos)
            positions.append(pos)
            break

        pos = new_pos
        positions.append(pos)

        if not np.array_equal(pos, final_pos):
            reward = MOVE_REWARD
            total_reward += reward
            agent.update(reward, pos-a, a, pos)
        else:
            reward = GOAL_REWARD
            total_reward += reward
            agent.update(reward, pos-a, a, pos)
        
    return positions, total_reward

def render_agent_policy (agent, maze):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    fig,ax = plt.subplots(1,1, figsize=(8,8) )
    cmap = ListedColormap(["white", "black", "#dddddd", "red", "yellow"])
    im = ax.imshow(maze.cells, interpolation='none', aspect='auto', vmin=0, vmax=4, cmap=cmap)

    for i, j in np.ndindex(maze.cells.shape):
        pos = np.array((i,j))
        action = tuple(agent.next_action(pos))
        if action == (-1,0):
            t = r"$\uparrow$"
        elif action == (1,0):
            t = r"$\downarrow$"
        elif action == (0,1):
            t = r"$\rightarrow$"
        elif action == (0,-1):
            t = r"$\leftarrow$"
        text = ax.text(j, i, t, ha="center", va="center", color="b", fontsize=32)

    fig.tight_layout()
    plt.show()

def render_agent_Q (agent, maze):
    fig,ax = plt.subplots(1,1, figsize=(8,8) )
    cmap = ListedColormap(["white", "black", "#dddddd", "red", "yellow"])
    im = ax.imshow(maze.cells, interpolation='none', aspect='auto', vmin=0, vmax=4, cmap=cmap)

    for i, j in np.ndindex(maze.cells.shape):
        pos = np.array((i,j))
        v = max([agent.getQ(pos, a) for a in actions])
        text = ax.text(j, i, v, ha="center", va="center", color="b", fontsize=32)

    fig.tight_layout()
    plt.show()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', action='store', required=False, default="qlearning", type=str)
    parser.add_argument('--episodes', action='store', required=False, default=10, type=int)
    parser.add_argument('--size', action='store', required=False, default=5, type=int)
    parser.add_argument('--reward_model', action='store', required=False, default=0, type=int)
    parser.add_argument('--norender', action='store_true', required=False, default=False)
    parser.add_argument('--random_initial_pos', action='store_true', required=False, default=False)
    parser.add_argument('--slippery_cells', action='store', required=False, default=0, type=int)
    parser.add_argument('--action_selection', action='store', required=False, default="softmax")
    parser.add_argument('--plot_reward', action='store_true', required=False, default=False)
    parser.add_argument('--seed', action='store', required=False, default=1, type=int)


    args = parser.parse_args()

    np.random.seed(args.seed)

    size=(args.size,args.size)
    final_pos = np.array([0, size[1]-1])

    i,j=(0,1)
    blocked_cells=[]
    while i < size[0]-1 and j<size[1]-1:
        blocked_cells.append((i,j))
        i+=1
        j+=1

    slippery_cells=[]
    assert(args.slippery_cells >= 0)
    for i in range(args.slippery_cells):
        c = (np.random.randint(args.size), np.random.randint(args.size))
        if c != tuple(final_pos):
            slippery_cells.append(c)
        
    m = Maze(size, blocked_cells, slippery_cells)


    if args.agent == "random":
        agent = RandomAgent(m)
    if args.agent == "mdp":
        assert(args.reward_model == 0)
        agent = MDPBasedAgent(m)
    elif args.agent == "qlearning":
        agent = QlearningAgent(m, action_selection_policy=args.action_selection)
    elif args.agent == "sarsa":
        agent = SARSAAgent(m, action_selection_policy=args.action_selection)
        
    # train
    if args.agent != "mdp":
        rewards=np.zeros((args.episodes,))
        for episode in range(args.episodes):
            _, reward = run_episode(m, agent, final_pos,
                    random_initial_pos=args.random_initial_pos,
                    reward_model=args.reward_model)

            rewards[episode] = reward

        print(f"Avg. reward: {np.mean(rewards)}")
        if args.plot_reward:
            fig,ax = plt.subplots(1,1)
            cum_avg = np.cumsum(rewards)/np.arange(1,1+len(rewards))
            ax.plot(cum_avg)
            plt.show()

    # -----------------
    agent.action_selection_policy = ASP_GREEDY
    history, reward = run_episode(m, agent, final_pos,
            random_initial_pos=args.random_initial_pos,
            reward_model=args.reward_model)
    print(f"Final reward: {reward}")
    if not args.norender:
        if args.agent != "mdp":
            title = f"After {args.episodes} episodes"
        else:
            title = "Optimal MDP policy"
        m.render(history, final_pos, title=title)

    #render_agent_policy(agent, m)
    #render_agent_Q(agent, m)


