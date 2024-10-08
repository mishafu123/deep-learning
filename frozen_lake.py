import numpy as np
import gym
import random
import time
from IPython.display import clear_output
import warnings

#ideally implemented in jupyter notebook
warnings.filterwarnings("ignore", category=DeprecationWarning) #b/c of version issue ignore warning

env = gym.make("FrozenLake-v1")

action_space_size = env.action_space.n #columns
state_space_size =  env.observation_space.n #rows

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

#algorithm parameters
num_episodes = 10000 #num to play during training
max_steps_per_episode = 100 #number of steps agent is allowed to take per episode, terminate with 0 points

learning_rate = 0.1  #alpha
discount_rate = 0.99 #gama

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001 #try 0.01

rewards_all_episodes = []

for episode in range(num_episodes):
    state, _ = env.reset() #reset state of environment back to starting state

    done = False #whether or not episode is over
    rewards_current_episode = 0 #rewards, only 1 when getting to end point

    for step in range(max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0,1) #set random value between 0 and 1, determine exploit vs explore
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :]) #take the action that has the highest value in q value, aka exploit
        else:
            action = env.action_space.sample() #explore, take an action randomly/sample

        new_state, reward, done, truncated, info = env.step(action) #take action

        #Update Q-talbe for Q(s,a), ased on formula of updating q-learning table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        #update state and reward
        state = new_state
        rewards_current_episode += reward

        #check if fell into a hole, reached limit, or reached the end
        if done == True or truncated:
            break

    #exploration rate change
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    #append reward to list of rewards from all episodes
    rewards_all_episodes.append(rewards_current_episode)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

print("\n\n********Q-table********\n")
print(q_table)

#0.700 means winning 70% of the time



#SFFF
#FHFH
#FFFH
#HFFG