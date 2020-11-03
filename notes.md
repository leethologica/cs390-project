# Genetic Algorithms

---

## Index

1. [Introduction](#introduction)
2. [Common Problems](#common-problems)
3. [Potential Project Ideas](#project-ideas)
4. [Resources](#resources)

---

## Introduction <a name="introduction"></a>

Genetic algorithms are a type of learning algorithm that is founded on the idea that crossing over the weights of two good neural networks should result in a better neural network.

##### Basic Intuition

* Sort of mimicking evolution
  * Produce a basis for "natural selection"
    * A population of networks is generated, all of which are competing against each other
  * Spread the genes of high-performing networks
    * Also sprinkle in some random mutations
  * Produce "generations" of the network, with the hope that each generation takes the good qualities of its ancestors and improves them, then spreads them to its descendants

##### General Idea of Implementation

* A set of random weights are generated
  * This network is called the "first agent"
* A set of tests are performed on the agent, and is scored
* Repeat this several times to create a population
* Select s subset of high performing agents in this population to be used for crossover
  * Every time there is a crossover, there is a small chance of **mutation**, that is, a random value that is in neither of the parent's weights
* Continually repeat this process

##### Pros:

* Relatively computationally inexpensive
  * No linear algebra of back-propagation
* Adaptable
  * Genetic algorithms can be applied to many different problems (classifiers, GANs, AI)

##### Cons:

* Takes a long period of time
  * Unlucky crossovers and mutations can worsen the network's performance and make it harder to reach convergence. However, these mutations could also potentially improve the network

---

## Common Problems

### Poor Convergence (Stuck at Local Extrema)

* Network gets stuck at a poor solution and does not change
  * This can happen if the population size of each generation is too small (low genetic diversity)
  * This can also happen if there is no mutation to each generation
    * Adding a mutation chance will help explore more of the solution space

### Lack of Convergence

* Network does not converge at all or takes an unpractical time to reach convergence
  * May be too much mutation happening

---

## Potential Project Ideas <a name="project-ideas"></a>

* Snake game (there is a fully coded example online so probably not exactly this)
* Wifi dinosaur game
* Pacman
* Pong

---

## Resources

* Introduction to Genetic Algorithms and General Intuition
  * https://towardsdatascience.com/using-genetic-algorithms-to-train-neural-networks-b5ffe0d51321
  * https://www.youtube.com/watch?v=XP8R0yzAbdo
* Fully programmed PacMan in python
  * https://github.com/greyblue9/pacman-python

