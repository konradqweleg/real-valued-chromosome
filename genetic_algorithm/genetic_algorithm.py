import logging

import numpy as np
from app.real_value_chromosome import RealValuedChromosome
import matplotlib.pyplot as plt


def setup_logger(log_level=logging.INFO):
    with open('app.log', 'w'):
        pass
    logger = logging.getLogger()
    logger.setLevel(log_level)

    file_handler = logging.FileHandler('app.log')
    console_handler = logging.StreamHandler()

    file_handler.setLevel(log_level)
    console_handler.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def set_log_level(log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)


setup_logger()
set_log_level(logging.INFO) #DEBUG OR INFO


class GeneticAlgorithm:
    def __init__(self, num_parameters, gene_range, population_size, num_iterations, fitness_function, selection_method, crossover_method, mutation_method,
                 optimization_type='minimization', percentage_best_to_transfer = 0.2, crossover_probability = 0.5):
        self.logger = logging.getLogger(__name__)

        self.population = [RealValuedChromosome(num_parameters, gene_range=gene_range) for _ in range(population_size)]
        self.fitness_function = fitness_function
        self.optimization_type = optimization_type
        self.num_iterations = num_iterations
        self.population_size = population_size

        self.selection_method = selection_method

        self.crossover_method = crossover_method

        self.mutation_method = mutation_method
        self.crossover_probability = crossover_probability

        self.avg_fitness_values = []
        self.std_fitness_values = []
        self.min_fitness_values = []

        self.percentage_best_to_transfer = percentage_best_to_transfer

        self.logger.info(f"Genetic algorithm initialized with population size: {population_size}, number of parameters: {num_parameters}, optimization type: {optimization_type}, gene range: {gene_range}, num iterations: {num_iterations}, selection method: {selection_method}, crossover method: {crossover_method}, mutation method: {mutation_method}, crossover probability: {crossover_probability}")

    def __str__(self):
        return f"Population: {[str(chromosome) for chromosome in self.population]}"

    def find_best_chromosome(self):
        fitness_values = [self.fitness_function.calculate(chromosome.genes) for chromosome in self.population]
        if self.optimization_type == 'minimization':
            best_chromosome_index = fitness_values.index(min(fitness_values))
        else:
            best_chromosome_index = fitness_values.index(max(fitness_values))
        return self.population[best_chromosome_index]

    def find_best_fitness_value(self):
        fitness_values = [self.fitness_function.calculate(chromosome.genes) for chromosome in self.population]
        if self.optimization_type == 'minimization':
            return min(fitness_values)
        else:
            return max(fitness_values)

    def run(self):
        for iteration in range(self.num_iterations):
            fitness_values = [self.fitness_function.calculate(chromosome.genes) for chromosome in self.population]

            avg_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)
            self.avg_fitness_values.append(avg_fitness)
            self.std_fitness_values.append(std_fitness)
            self.min_fitness_values.append(min(fitness_values))

            num_best_to_transfer = int(self.population_size * self.percentage_best_to_transfer)
            best_individuals = sorted(self.population, key=lambda x: self.fitness_function.calculate(x.genes),
                                      reverse=(self.optimization_type == "maximization"))[:num_best_to_transfer]

            index_offset_caused_by_numeration_from_zero = 1
            self.logger.debug(
                f"Iteration {iteration + index_offset_caused_by_numeration_from_zero}: Fitness values: {fitness_values}")

            chromosomes_values = [chromosome.genes for chromosome in self.population]
            self.logger.debug(f"Chromeosomes values: {chromosomes_values}")

            selected_population = self.selection_method.select(self.population, fitness_values, self.optimization_type)

            self.logger.debug(f"Selected population: {[str(chromosome) for chromosome in selected_population]}")

            expected_population_size = self.population_size - len(best_individuals)

            crossover_population = self.crossover_method.crossover(selected_population, expected_population_size, self.crossover_probability)

            self.logger.debug(f"Crossover population: {[str(chromosome) for chromosome in crossover_population]}")

            mutated_population = self.mutation_method.mutate(crossover_population)

            self.logger.debug(f"Mutated population: {[str(chromosome) for chromosome in mutated_population]}")

            self.population = mutated_population

            self.population = mutated_population + best_individuals


            if iteration % 100 == 0:
                best_fitness_value_in_iteration = self.find_best_fitness_value()
                self.logger.info(f"Best fitness value in iteration: {best_fitness_value_in_iteration}")
                self.logger.info(f"Number of iteration: {iteration}")



        best_chromosome = self.find_best_chromosome()
        fitness_value_of_best_chromosome = self.fitness_function.calculate(best_chromosome.genes)
        self.logger.info(f"Best chromosome: {best_chromosome}")
        self.logger.info(f"Fitness value of the best chromosome: {fitness_value_of_best_chromosome}")

        return best_chromosome, fitness_value_of_best_chromosome, self.avg_fitness_values, self.std_fitness_values, self.min_fitness_values





# the_best_selection = BestSelection(percentage_the_best_to_select=0.5)
# arithmetic_crossover = ArithmeticCrossover(alpha=0.5, gene_range=(-500, 500))
# random_resetting = RandomResetting(mutation_rate=0.01, gene_range=(-500, 500))
#
# ga = GeneticAlgorithm(num_parameters=2, gene_range=(-500, 500), population_size=10, num_iterations=10000,
#                       fitness_function=Szwefel(), selection_method=the_best_selection,crossover_method=arithmetic_crossover, mutation_method=random_resetting,  optimization_type= 'minimization')
# ga.run()
