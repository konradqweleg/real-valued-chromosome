import benchmark_functions as bf
from app.function.function_to_calculate import FunctionToCalculate

class Szwefel(FunctionToCalculate):
    def calculate(self, variables):
        n = len(variables)
        func = bf.Schwefel(n_dimensions=n)
        return func(variables)
