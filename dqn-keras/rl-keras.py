import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from collections import deque

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = 0.05

        self.model        = self.create_model()
        self.target_model = self.create_model()


    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        #model.add(Dense(24, input_dim=4, activation="relu"))
        model.add(Dense(24, activation='relu', input_shape=(4,)))
        #model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model


    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        Q = self.model.predict(state)
        return np.argmax(Q[0])


    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state.reshape((1, 4)),
                            action,
                            reward,
                            new_state.reshape((1, 4)),
                            done])


    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in samples:
            if done:
                update = reward
            else:
                predictedReward = np.amax(self.model.predict(new_state)[0])
                update = (reward + self.gamma * predictedReward)
            Qval = self.model.predict(state)
            Qval[0][action] = update
            self.model.fit(state, Qval, verbose = 0)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)


    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)


    def save_model(self, fn):
        self.model.save(fn)


def main():
    env     = gym.make("CartPole-v0")
    env._max_episode_steps = 500
    #gamma   = 0.9
    #epsilon = .95

    #print(f"Shape of environment is {str(env.observation_space.shape[0])}")
    #exit()

    trials  = 500
    trial_len = 250

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    scores_arr = []
    for trial in range(trials):
        print(f"Performing trial {trial}")
	    #print("Yikes!")
        cur_state = env.reset()
        done = False
        score = 0
        #for step in range(trial_len):
        while not done:
            #print(f"On step {trial}")
            #env.render()
            action = dqn_agent.act(cur_state.reshape((1, 4)))
            new_state, reward, done, _ = env.step(action)

            reward = reward if not done else -20
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            #dqn_agent.replay()       # internally iterates default (prediction) model
            #dqn_agent.target_train() # iterates target model

            cur_state = new_state
            score += 1
        print(f"Trial {trial} has score {score}")
        scores_arr.append(score)
        avg_100 = 0
        if trial >= 101:
            list_100 = scores_arr[-100:]
            avg_100 = sum(list_100) / 100
            print(f"\tAverage over last 100 runs: {avg_100}")
        dqn_agent.replay()
    print("Saving high score")
    dqn_agent.save_model("highscore_model")
    print(scores_arr)

if __name__ == "__main__":
    main()
