import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
import gym

import time


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


def train_model(env, model, epochs, learning_rate=0.003, gamma=0.99, mem_size=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    score = []
    for epoch in range(epochs):
        score.append(run_epoch(env, model, optimizer, gamma, mem_size))

        if epoch % 50 == 0 and epoch > 0:
            print('Trajectory {}\tAverage Score: {:.2f}'.format(epoch, np.mean(score[-50:-1])))
    return score


def run_epoch(env, model, optimizer, gamma, mem_size=500):
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

    return len(transitions)


def build_model(env):
    in_dim = 1
    for i in env.observation_space.shape:
        in_dim *= i
    return nn.Sequential(
        nn.Linear(in_dim, 24),
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


def run_training_batch(template, env, epochs=500, clip=50):
    model = template(env)
    scores = train_model(env, model, epochs)
    return model, np.mean(scores[(-1 * clip):])


'''
    creates a pool of agents and returns the best model found
'''
def run_pool(env, template, pool_size=25):
    best_model = None
    best_score = 0
    for _ in range(pool_size):
        model, score = run_training_batch(template, env)
        if score > best_score:
            best_model = model
    return best_model


def main():
    env = gym.make('CartPole-v0')
    model = run_pool(env, build_model)
    input("training done")
    test_model(env, model, delay=0.05)


def oldmain():
    env = gym.make('CartPole-v0')
    # best_model = None
    model = build_model(env)
    train_model(env, model, 500)
    input("training done")
    test_model(env, model, delay=0.01)
    env.close()


if __name__ == '__main__':
    main()
