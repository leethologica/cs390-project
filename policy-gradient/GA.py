import os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.style.use('seaborn')

NUM_GENERATIONS = 200
POPULATION_SIZE = 100
SELECTION_SIZE = 15
MUTATION_RATE = 0.1
MUTATION_WEIGHT_RANGE = (-1, 1)
MUTATION_BIAS_RANGE = (-0.1, 0.1)

class GA:
  def __init__(self,
               model_builder,
               num_generations=NUM_GENERATIONS,
               population_size=POPULATION_SIZE,
               selection_size=SELECTION_SIZE,
               mutation_rate=MUTATION_RATE,
               mutation_weight_range=MUTATION_WEIGHT_RANGE,
               mutation_bias_range=MUTATION_BIAS_RANGE,
               crossover_type='uniform',
               descending_fitness=False,
               elitism=True
              ):
    self.num_generations = num_generations
    self.population_size = population_size
    self.selection_size = selection_size
    self.mutation_rate = mutation_rate
    self.mutation_weight_range = mutation_weight_range
    self.mutation_bias_range = mutation_bias_range
    self.model_builder = model_builder
    self.descending_fitness = descending_fitness
    self.elitism = elitism
    if crossover_type.lower() == 'uniform':
      self.crossover_method = self.__uniform_crossover
    elif 'single' in crossover_type.lower():
      self.crossover_method = self.__single_point_crossover
    else:
      raise(ValueError('Unknown crossover method %s' % crossover_type))
      return None
    self.current_population = None
    self.builder_args = list()

    # Saving and logging purposes
    self.generation_number = 1
    self.generation_fitness = list()
    self.top_generation_fitness = list()


  '''
  Initializes the seed population using model_builder

  Arguments:
    *args: any arguments necessary to be passed into model_builder
  Returns: the initialized population
  '''
  def initialize_population(self, *args):
    self.builder_args = [*args]
    self.current_population = [self.model_builder(*self.builder_args) for _ in range(self.population_size)]
    return self.current_population


  '''
  Performs the genetic algorithm on the seeded population.

  Arguments:
    evaluator: the function used to obtain fitness values for each individual
    **kwargs: any relevant arguments passed into the evaluator
  Returns: the most fit individual in the population after performing GA on num_generations generations
  '''
  def get_best_model(self, evaluator, verbose=False, **kwargs):
    self.verbose=verbose
    fitness = None
    for generation in range(self.generation_number, self.num_generations + 1):
      fitness = list()
      if self.verbose:
        start = time.time()
        print('--------------------------------\n'\
              '----------GENERATION %s----------\n'\
              '--------------------------------\n' % generation)

      i = 1
      for individual in self.current_population:
        if self.verbose:
          print('----------INDIVIDUAL %s----------' % i)
        fitness.append(evaluator(individual, *kwargs['params']))
        i+=1
      self.current_population = self.__evolve(fitness)
      self.generation_number += 1
      if self.verbose:
        end = time.time()
        print('Generation produced in %ds' % (end - start))
      self.__log_fitness(sorted(fitness, reverse=self.descending_fitness))


    sorted_population = self.__sort_population(self.current_population, fitness)
    self.__save_mating_pool(sorted_population[:self.selection_size])
    self.__graph_fitness()
    return sorted_population[0]


  '''
  Evolves the current population to obtain the next generation of individuals.

  Arguments:
    fitness: the list of fitnesses for each individual in the current population
  Returns: the new generation
  '''
  def __evolve(self, fitness):
    sorted_population = self.__sort_population(self.current_population, fitness)

    new_population = self.__crossover(sorted_population[:self.selection_size])
    self.__save_mating_pool(sorted_population[:self.selection_size])
    return new_population


  '''
  Performs crossover and mutation to produce a new population of offspring.

  Arguments:
    parents: the mating pool from which to create offspring from
  Returns: the new offspring population
  '''
  def __crossover(self, parents):
    new_population = list()
    start = 0
    # Elitism - most fit parent is copied to next generation
    if self.elitism:
      new_population.append(parents[0])
      start = 1

    for i in range(start, self.population_size):
      # Select parents for new individual
      parent_A, parent_B = random.sample(parents, 2)

      # Construct new individual
      dense_layer_indexes = [x for x in range(len(parent_A)) if isinstance(parent_A[x], nn.Linear)]
      individual = self.model_builder(*self.builder_args)
      for j in dense_layer_indexes:
        parent_weights = (parent_A[j].weight, parent_B[j].weight)
        parent_biases = (parent_A[j].bias, parent_B[j].bias)
        weights, biases = self.crossover_method(parent_weights, parent_biases)

        individual.state_dict()['%s.weight' % j][:] = torch.Tensor(weights)
        individual.state_dict()['%s.bias' % j][:] = torch.Tensor(biases)
      new_population.append(individual)
    return new_population


  '''
  Helper function to perform uniform crossover.

  Arguments:
    parent_weights: a tuple of matrix weights for each of the two parents being mated
    parent_biases: a tuple of bias lists for each of the two parents being mated
  Returns: a tuple of the new weights and biases for this offspring
  '''
  def __uniform_crossover(self, parent_weights, parent_biases):
    weights = np.zeros(parent_weights[0].shape)
    for r in range(weights.shape[0]):
      for c in range(weights.shape[1]):
        weights[r][c] = (parent_weights[0][r][c] if random.random() < 0.5 else
                         parent_weights[1][r][c])
        # Add random mutations
        if random.random() < self.mutation_rate:
          weights[r][c] += random.uniform(*self.mutation_weight_range)
    biases = np.zeros(parent_biases[0].shape)
    for k in range(biases.shape[0]):
      biases[k] = (parent_biases[0][k] if random.random() < 0.5 else
                   parent_biases[1][k])
      # Add random mutations
      if random.random() < self.mutation_rate:
        biases[k] += random.uniform(*self.mutation_bias_range)
    return (weights, biases)


  '''
  Helper function to perform single point crossover.

  Arguments:
    parent_weights: a tuple of matrix weights for each of the two parents being mated
    parent_biases: a tuple of bias lists for each of the two parents being mated
  Returns: a tuple of the new weights and biases for this offspring
  '''
  def __single_point_crossover(self, parent_weights, parent_biases):
    weights = np.zeros(parent_weights[0].shape)
    weight_partition = random.randint(0, parent_weights[0].shape[0] * parent_weights[0].shape[1])
    i = 0
    for r in range(weights.shape[0]):
      for c in range(weights.shape[1]):
        if i < weight_partition:
          weights[r][c] = parent_weights[0][r][c]
        else:
          weights[r][c] = parent_weights[1][r][c]
        # Add random mutations
        if random.random() < self.mutation_rate:
          weights[r][c] += random.uniform(*self.mutation_weight_range)
        i += 1
    biases = np.zeros(parent_biases[0].shape)
    bias_partition = weight_partition // weights.shape[0]
    for k in range(biases.shape[0]):
      if k < bias_partition:
        biases[k] = parent_biases[0][k]
      else:
        biases[k] = parent_biases[1][k]
      # Add random mutations
      if random.random() < self.mutation_rate:
        biases[k] += random.uniform(*self.mutation_bias_range)
    return (weights, biases)


  '''
  Helper function to sort a population by order of fitness.

  Arguments:
    population: the population to be sorted
    fitness: the list of fitnesses for each individual in the population to be sorted
  Returns: the sorted population
  '''
  def __sort_population(self, population, fitness):
    # Order the population in order of fitness
    r = -1 if self.descending_fitness else 1
    sorted_indexes = sorted(range(len(fitness)), key=lambda x: r * fitness[x])
    if self.verbose:
      print('Selected individuals: %s' % [i + 1 for i in sorted_indexes[:self.selection_size]])
    sorted_population = [population[i] for i in sorted_indexes]
    return sorted_population


  '''
  Saves the current mating pool to speed up future training or to recover from crashes

  Arguments:
    mating_pool: the matining pool to be saved
  Returns: None
  '''
  def __save_mating_pool(self, mating_pool):
    i = 0
    for individual in mating_pool:
      save_path = 'saved-mating-pool/parent%s.pt' % i
      torch.save(individual.state_dict(), save_path)
      i += 1


  '''
  Logs the average fitness of the current population and the current mating pool to an
  output file saved to the "saved-fitness" folder in the current directory. Also keeps
  a record of these logged values throughout the duration of the genetic algorithm.

  Arguments:
    fitness: the fitness of each individual in the current population
  Returns: None
  '''
  def __log_fitness(self, fitness):
    average_fitness = np.mean(fitness)
    average_top_fitness = np.mean(fitness[:self.selection_size])
    self.generation_fitness.append(average_fitness)
    self.top_generation_fitness.append(average_top_fitness)
    if self.verbose:
      print('Average fitness of generation: %f' % (average_fitness))
      print('    Average fitness of top %2d: %f' % (self.selection_size, average_top_fitness))
    with open('saved-fitness/generation-fitness.npy', 'wb') as f:
      np.save(f, self.generation_fitness)
    with open('saved-fitness/top-generation-fitness.npy', 'wb') as f:
      np.save(f, self.top_generation_fitness)


  '''
  Graphs the fitness of every generation throughout the entire genetic algorithm process.
  Saves this graph into a folder called "logs" in the current directory.

  Returns: None
  '''
  def __graph_fitness(self):
    df = pd.DataFrame({'generation': [i for i in range(1, self.generation_number)],
                       'fitness': self.generation_fitness,
                       'top': self.top_generation_fitness})
    output_path = 'logs/learning-curve.png'
    plt.figure(figsize=(10, 10))
    plt.plot('generation', 'fitness', data=df, color='red', label='Population')
    plt.plot('generation', 'top', data=df, color='blue', label='Top %s Individuals' % self.selection_size)
    plt.grid()
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness Over Time')
    plt.legend()
    plt.savefig(output_path)
    if self.verbose:
      print('Saved learning curve to %s' % output_path)
    plt.show()

