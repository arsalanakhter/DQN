# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
import numpy as np
import gym
import random
import os
try:
    os.chdir(os.path.join(os.getcwd(), '..\..\DQN'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# # DQN On Cartpole
# %% [markdown]
# Taken from https://github.com/jonkrohn/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
# %% [markdown]
# #### Processing Devices

# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %% [markdown]
# #### Dependencies

# %%

# %% [markdown]
# #### Set Hyper-parameters

# %%
env = gym.make('CartPole-v0')


# %%
state_size = env.observation_space.shape[0]
print(state_size)

action_size = env.action_space.n
print(action_size)

batch_size = 32
n_episodes = 1001
output_dir = 'model_output/cartpole'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %% [markdown]
# #### Define Agent

# %%
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        # Some Hyper Parameters below

        # Discount factor gamma
        self.gamma = 0.95
        # Exploration rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        # Step size for stochastic gradient descent optimizer(Adam)
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        # Define our dense neural network for approimating Q*(s,a)
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        # Output layer. Equal to the output size that we want. We also want the
        # activation to be linear because we do not want some abstracted output
        # instead, we want the direct output of actions.
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        # For remembering the episodes. I think it is used in experience replay
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Figure out what action to take given the state
        # Randomly pick a number. If it is less than epsilon, explore
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Otherwise, exploit
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = DQNAgent(state_size, action_size)


# %% [markdown]
# #### Interact with the environment
done = False
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(5000):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print('episode: {}/{}, score: {}, e: {:.2}'.format(e,
                                                               n_episodes, time, agent.epsilon))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + "weights_" +
                   '{:04d}'.format(e) + ".hdf5")
