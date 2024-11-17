import logging
import random
from app.selection.selection_method import SelectionMethod

class RouletteWheelSelection(SelectionMethod):
    def __init__(self, percentage_the_best_to_select):
        self.percentage_chromosomes_to_select = percentage_the_best_to_select
        self.logger = logging.getLogger(__name__)

    def select(self, population, fitness_scores, optimization_type='maximization'):
        self.logger.debug("Selecting method [roulette wheel selection]")
        self.logger.debug("Chromosomes: " + str([str(chromosome) for chromosome in population]))
        self.logger.debug("Fitness scores: " + str(fitness_scores))

        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            fitness_scores = [score - min_fitness for score in fitness_scores]

        self.logger.debug("Adjusted fitness scores (after handling negatives): " + str(fitness_scores))

        if optimization_type == 'minimization':
            max_fitness = max(fitness_scores)
            fitness_scores = [max_fitness - score for score in fitness_scores]

        total_fitness = sum(fitness_scores)
        adjustment = 0.01 * total_fitness
        fitness_scores = [score + adjustment for score in fitness_scores]
        total_fitness = sum(fitness_scores)
        self.logger.debug("Fitness scores: " + str(fitness_scores))

        if total_fitness == 0:
            self.logger.debug("Total fitness is zero. Assigning equal probabilities to all chromosomes.")
            relative_fitness_scores = [1 / len(population)] * len(population)
        else:
            relative_fitness_scores = [fitness_score / total_fitness for fitness_score in fitness_scores]

        self.logger.debug("Relative fitness scores: " + str(relative_fitness_scores))
        sum_relative_fitness_scores = sum(relative_fitness_scores)
        self.logger.debug("Sum of relative fitness scores: " + str(sum_relative_fitness_scores))

        selected_chromosomes = []
        for _ in range(int(len(population) * self.percentage_chromosomes_to_select)):
            r = random.random()
            cumulative_probability = 0
            for i in range(len(population)):
                cumulative_probability += relative_fitness_scores[i]
                if r <= cumulative_probability:
                    selected_chromosomes.append(population[i])
                    break

        return selected_chromosomes

    def __str__(self):
        return "Roulette wheel selection"