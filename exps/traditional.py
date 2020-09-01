from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

# Parity CNF (XOR).
formula = WCNF()
formula.append([1, 2, -3])
formula.append([1, -2, 3])
formula.append([-1, 2, 3])
formula.append([-1, -2, -3])

solver = RC2(formula, verbose=0)

for m in solver.enumerate():
    print('model {0} has cost {1}'.format(m, solver.cost))
