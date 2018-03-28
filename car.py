import gym
import numpy as np
import random

"""QLEARN CLASS NOT MY CODe"""
class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma, q={}):
        self.q = q
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get(str(state)+ str(action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get(str(state) + str(action), None)
        if oldv is None:
            self.q[str(state)+ str(action)] = reward
        else:
            self.q[str(state)+ str(action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

EPSILON=.05
ALPHA=.9
GAMMA=.2

env = gym.make("BinaryCarRacing-v0")
env.reset()

action_list = []
individual_actions = [
    "steer_left",
    "steer_right",
    "gas_on",
    "gas_off",
    "brake_on",
    "brake_off"
]

import itertools
for r in [1, 2, 3]:
    for comb in itertools.combinations(individual_actions, r=r):
        action_list.append(comb)

def assign_val(x, index, val):
    x[index] = val
    return x

action_effects = {
    "steer_left":     lambda x: assign_val(x, 0, -1),
    "steer_straight": lambda x: assign_val(x, 0, 0),
    "steer_right":    lambda x: assign_val(x, 0, 1),
    "gas_on":         lambda x: assign_val(x, 1, 1),
    "gas_off":        lambda x: assign_val(x, 1, 0),
    "brake_on":       lambda x: assign_val(x, 2, 1),
    "brake_off":      lambda x: assign_val(x, 2, 0)
}

import json as jason
agent = QLearn(
    actions=action_list,
    epsilon=EPSILON,
    alpha=ALPHA,
    gamma=GAMMA,
    q=jason.load(open("./agent_q.agent"))
)

MAX_READING = 10
import math
def convert_to_discrete(reading, max_reading):
    if reading == 0 or reading == -1:
        return -1
    return math.floor(reading/max_reading * 5)

for iteration in range(500):
    action = np.array([0, 0, 0])
    action_chosen = None
    state = None
    step_reward = None
    done = False
    while not done:
        new_state, step_reward, done, _ = env.step(action)
        new_state = list(map(lambda x: convert_to_discrete(x, MAX_READING), new_state))
        non_negative = False
        for s in new_state:
            if s != -1:
                non_negative = True
        if not non_negative:
            done = True
        if state:
            agent.learn(state, action_chosen, step_reward, new_state)
        action_chosen = agent.chooseAction(new_state)

        for act in action_chosen:
            action = action_effects[act](action)
        state = new_state
        if iteration % 50 == 0:
            env.render()
    env.reset()
    if iteration % 10 == 0:
        with open("./agent_q.agent", "w+") as f:
            f.write(jason.dumps(agent.q))
        print("agent_q.agent written")
