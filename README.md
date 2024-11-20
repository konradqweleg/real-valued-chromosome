# Genetic Algorithm Implementation

This project implements a genetic algorithm for optimization. The genetic algorithm is a search heuristic that mimics the process of natural selection. This implementation includes various components such as selection methods, crossover methods, and mutation methods.

## Prerequisites

- Python 3.x
- Required Python packages: `numpy`, `matplotlib`, `logging`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/konradqweleg/genetic-algorithm.git
    cd genetic-algorithm
    ```

2. Install the required packages:
    ```sh
    pip install numpy matplotlib
    ```

## License
This project is licensed under the MIT License. 
   
## Genetic Algorithm Class

The GeneticAlgorithm class implements the genetic algorithm. It includes methods for initialization, finding the best chromosome, finding the best fitness value, and running the algorithm.  
Initialization
The constructor initializes the genetic algorithm with the following parameters:
* num_parameters: Number of parameters in each chromosome.
* gene_range: Range of gene values.
* population_size: Size of the population.
* num_iterations: Number of iterations to run the algorithm.
* fitness_function: Function to evaluate the fitness of chromosomes.
* selection_method: Method for selecting chromosomes for reproduction.
* crossover_method: Method for crossing over chromosomes.
* mutation_method: Method for mutating chromosomes.
* optimization_type: Type of optimization (minimization or maximization).
* percentage_best_to_transfer: Percentage of the best individuals to transfer to the next generation.
* crossover_probability: Probability of crossover.

### Methods
* __str__: Returns a string representation of the population.
* find_best_chromosome: Finds the best chromosome in the population.
* find_best_fitness_value: Finds the best fitness value in the population.
* run: Runs the genetic algorithm for the specified number of iterations.



### Example Usage

```python
from app.genetic_algorithm import GeneticAlgorithm
from app.selection_methods import BestSelection
from app.crossover_methods import ArithmeticCrossover
from app.mutation_methods import RandomResetting
from app.fitness_functions import Schwefel

# Define the selection, crossover, and mutation methods
the_best_selection = BestSelection(percentage_the_best_to_select=0.5)
arithmetic_crossover = ArithmeticCrossover(alpha=0.5, gene_range=(-500, 500))
random_resetting = RandomResetting(mutation_rate=0.01, gene_range=(-500, 500))

# Initialize the genetic algorithm
ga = GeneticAlgorithm(
    num_parameters=2,
    gene_range=(-500, 500),
    population_size=10,
    num_iterations=10000,
    fitness_function=Schwefel(),
    selection_method=the_best_selection,
    crossover_method=arithmetic_crossover,
    mutation_method=random_resetting,
    optimization_type='minimization'
)

# Run the genetic algorithm
best_chromosome, best_fitness_value, avg_fitness_values, std_fitness_values, min_fitness_values = ga.run()



