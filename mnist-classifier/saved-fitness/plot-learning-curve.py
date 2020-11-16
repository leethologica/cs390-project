import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAVED_FITNESS_LOGS = sys.argv[1]

with open(SAVED_FITNESS_LOGS + '/generationFitness.npy', 'rb') as f:
  generationFitness = list(np.load(f))
with open(SAVED_FITNESS_LOGS + '/topGenerationFitness.npy', 'rb') as f:
  topGenerationFitness = list(np.load(f))

df = pd.DataFrame({'generation': [i + 1 for i in range(len(generationFitness))],
                   'fitness': generationFitness,
                   'top': topGenerationFitness})
plt.figure(figsize=(10,10))
plt.plot('generation', 'fitness', data=df, color='red', label='Population')
plt.plot('generation', 'top', data=df, color='blue', label='Top 15 Individuals')
plt.grid()
plt.xlabel('Generation')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Over Time')
plt.legend()
plt.show()
