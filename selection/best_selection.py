from app.selection.selection_method import SelectionMethod
import logging


class BestSelection(SelectionMethod):

    def __init__(self, percentage_the_best_to_select=0.5):
        self.percentage_the_best_to_select = percentage_the_best_to_select
        self.logger = logging.getLogger(__name__)

    def select(self, population, fitness_scores, optimization_type='minimization'):
        self.logger.debug(f"Selecting the best chromosomes from the population, percentage the ebst to select: {self.percentage_the_best_to_select}")

        if optimization_type == 'minimization':
            selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        else:
            selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)


        num_selected = int(len(population) * self.percentage_the_best_to_select)
        selected_population = [population[i] for i in selected_indices[:num_selected]]
        return selected_population