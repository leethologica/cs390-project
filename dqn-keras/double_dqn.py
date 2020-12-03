import gym
import random
import os
import numpy as np
from collections      import deque
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Dense
from tensorflow.keras.optimizers import Adam

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "cartpole"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.99 #0.95
        self.exploration_rate   = 0.5 #1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.tau                = 0.1
        self.C                  = 1000
        self.online_model       = self._build_model()
        self.target_model       = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=4, activation='relu'))
        model.add(Dense(64, activation='relu'))
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self, fn):
            self.online_model.save(fn + "-online.h5")
            self.target_model.save(fn + "-target.h5")

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        state = np.reshape(state, (1, 4))
        act_values = self.online_model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.memory) <= 2000:
            self.memory.append(experience)
        else:
            self.memory[0] = event

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        minibatch = random.sample(self.memory, sample_batch_size)

        states = []
        Q_wants = []
        
        #Find updates
        for event in minibatch:
            state,action,reward,next_state,done = event
            states.append(state)
            
            #Find Q_target
            state_tensor = np.reshape(state,(1,4))  # keras takes 2d arrays
            Q_want = self.online_model.predict(state_tensor)[0]     # all elements of this, except the action chosen, stay
                                                             # the same                       
            
            #If state is terminal, Q_target(action) = reward
            if done == True:
                Q_want[action] = reward
            
            # Q_want(action) = reward + gamma*max_a Q_target(next_state, a*)  -- note I sample from the target network
            # where a* = argmax Q(next_state,a)
            # Doing this -- i.e. finding a* from the behavior network, is what 
            # distinguihses DDQN from DQN
            else:
                next_state_tensor = np.reshape(next_state,(1,4))  
                Q_next_state_vec = self.online_model.predict(next_state_tensor)
                action_max = np.argmax(Q_next_state_vec)
                
                Q_target_next_state_vec = self.target_model.predict(next_state_tensor)[0]
                Q_target_next_state_max = Q_target_next_state_vec[action_max]
                
                Q_want[action] = reward + self.gamma*Q_target_next_state_max
                Q_want_tensor = np.reshape(Q_want,(1,len(Q_want)))
                #self.model.fit(state_tensor,Q_want_tensor,verbose=False,epochs=1)
            
            Q_wants.append(Q_want)
            
        
        #Here I fit on the whole batch. Others seem to fit line-by-line
        #Dont' think (hope) it makes much difference
        states = np.array(states)
        Q_wants = np.array(Q_wants)
        self.online_model.fit(states,Q_wants,verbose=False, epochs=1)

    def update_target_model(self):
        q_network_theta = self.online_model.get_weights()
        target_network_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta,target_network_theta):
            target_weight = target_weight * (1-self.tau) + q_weight * self.tau
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_network_theta)

    def delayed_target_update(self, step):
        if step % self.C == 0:
            w = self.online_model.get_weights()
            self.target_model.set_weights(w)

    def update_exploration_rate(self):
        self.exploration_rate = max(self.exploration_rate*self.exploration_decay, self.exploration_min)

class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 5000
        self.env               = gym.make('CartPole-v1')

        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)


    def run(self):
        try:
            scores_list = []
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                while not done:
                    #self.env.render() #uncomment for play

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done) #comment for play
                    state = next_state
                    
                    if len(self.agent.memory) > 500:
                        self.agent.replay(self.sample_batch_size)
                        self.agent.update_target_model()

                    index += 1
                
                self.agent.update_exploration_rate()
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                scores_list.append(index + 1)
                #self.agent.replay(self.sample_batch_size) #comment for play
                #self.agent.update_target_model()
        finally:
            self.agent.save_model("rl-ddqn-f1")
            print(scores_list)

if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()