import frozen_lake
import numpy as np
import gym
import random
import time
from IPython.display import clear_output
import warnings

with open("frozen_lake.py") as f:
    exec(f.read())

env = gym.make("FrozenLake-v1", render_mode="human")
#ok training is complete, time to deploy it
#keep in mind on frozen lake/slippery you sometimes go somewhere you didn't intend!
for episode in range(3): #run 3 times
    state, _ = env.reset()
    done = False
    print("******EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1) #sleep for 1s to read

    for step in range(max_steps_per_episode):
        clear_output(wait = True)
        env.render()
        time.sleep(0.3)

        state = int(state)
        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, info = env.step(action)

        if done:
            clear_output(wait = True)
            env.render()
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
            clear_output(wait=True)
            break
        state = new_state
env.close()