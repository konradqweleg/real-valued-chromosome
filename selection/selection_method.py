from abc import ABC, abstractmethod


class SelectionMethod(ABC):
    @abstractmethod
    def select(self, population, fitness_scores, optimization_type='maximization'):
        """
        Wybiera chromosomy z populacji na podstawie ocen funkcji celu.

        Parametry:
        population (list): Lista chromosomów w populacji.
        fitness_scores (list): Lista ocen funkcji celu odpowiadających chromosomom.

        Zwraca:
        BinaryChromosome: Wybrane chromosomy.
        """
        pass
