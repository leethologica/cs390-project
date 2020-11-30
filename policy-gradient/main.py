import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
import gym

import time

from GA import GA


def explore(env, model, num_steps):
    env_state = env.reset()
    done = False
    transitions = []
    score = []
    action_space = np.array([i for i in range(env.action_space.n)])
    for t in range(num_steps):
        action_weights = model(torch.from_numpy(env_state).float())
        action = np.random.choice(action_space, p=action_weights.data.numpy())
        prev_state = env_state
        env_state, _, done, info = env.step(action)
        transitions.append((prev_state, action, t + 1))
        if done:
            break
    score.append(len(transitions))
    reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))
    return transitions, reward_batch


def train_model(model, env, epochs, clip, learning_rate=0.003, gamma=0.99, mem_size=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    score = []
    for epoch in range(1, epochs + 1):
        transitions, rewards = explore(env, model, mem_size)
        batch_Gvals = []
        for i in range(len(transitions)):
            new_Gval = 0
            power = 0
            for j in range(i, len(transitions)):
                new_Gval = new_Gval + ((gamma ** power) * rewards[j]).numpy()
            power += 1
            batch_Gvals.append(new_Gval)
        expected_returns_batch = torch.FloatTensor(batch_Gvals)
        expected_returns_batch /= expected_returns_batch.max()
        states = torch.Tensor([s for (s, a, r) in transitions])
        actions = torch.Tensor([a for (s, a, r) in transitions])
        preds = model(states)
        probabilities = preds.gather(dim=1, index=actions.long().view(-1, 1)).squeeze()
        loss = -torch.sum(torch.log(probabilities) * expected_returns_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        score.append(len(transitions))
        if epoch % 100 == 0 and epoch > 0:
            print('Trajectory {}\tAverage Score: {:.2f}'.format(epoch, np.mean(score[-50:])))
    return np.mean(score[(-1 * clip):])


def build_model(env):
    return nn.Sequential(
        nn.Linear(np.prod(env.observation_space.shape), 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, env.action_space.n),
        nn.Softmax(dim=0)
    )


def test_model(env, model, steps=100, delay=0.):
    model.eval()
    rewards = []
    env_state = env.reset()
    for i in range(steps):
        pred = model(torch.from_numpy(env_state).float())
        action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
        env_state, reward, done, _ = env.step(action)
        env.render()
        rewards.append(reward)
        if delay > 0:
            time.sleep(delay)
        if done:
            print("reward: ", sum([r for r in rewards]))
            return sum([r for r in rewards])


def main():
    env = gym.make('CartPole-v0')
    # For "quick" testing purposes use num_generations=2, population_size=3, selection_size=2
    # For actual testing purposes use something a lot higher, for example:
    # num_generations=100, population_size=200, selection_size=20
    ga = GA(num_generations=2,
            population_size=3,
            selection_size=2,
            mutation_weight_range=(-0.1, 0.1),
            mutation_bias_range=(-0.1, 0.1),
            descending_fitness=True,
            elitism=False, # switch to True to see if GA is more or less effective
            model_builder=build_model)
    ga.initialize_population(env)
    model = ga.get_best_model(train_model, verbose=True, params=[env, 500, 50])
    input("training done")
    test_model(env, model, delay=0.05)


if __name__ == '__main__':
    main()
