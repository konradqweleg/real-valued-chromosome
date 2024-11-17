from app.function.function_to_calculate import FunctionToCalculate
from opfunu.cec_based import F12014

class Cec2014F1(FunctionToCalculate):
    def calculate(self, variables):
        n = len(variables)
        func = F12014(ndim=n)
        return func.evaluate(variables)

