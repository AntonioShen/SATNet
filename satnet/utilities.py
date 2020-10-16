import torch
import math

from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

def pretty_print(**kwargs):
    for name, arg in kwargs.items():
        print('='*10, name, '='*10)

        # Vertical print of lists.
        if isinstance(arg, list):
            print('[')
            for item in arg:
                print('  ', item)
            print(']')

        else:
            print(arg)

def _3_sat_to_max_2_sat(S):
    # Using https://math.stackexchange.com/questions/1633005/how-exactly-does-a-max-2-sat-reduce-to-a-3-sat
    S = torch.FloatTensor(S)
    S_prime = torch.zeros([S.size()[0]*10, S.size()[1] + S.size()[0]])
    num_aux = len(S)

    for aux, (t, l1, l2, l3) in enumerate(S):
        aux_true = [0]*num_aux
        aux_true[aux] = 1
        aux_false = [0]*num_aux
        aux_true[aux] = -1
        aux_absent = [0]*num_aux

        clauses = torch.FloatTensor([
            [-1, l1, 0, 0] + aux_absent,
            [-1, 0, l2, 0] + aux_absent,
            [-1, 0, 0, l3] + aux_absent,
            [-1, 0, 0, 0] + aux_true,
            [-1, -l1, -l2, 0] + aux_absent,
            [-1, 0, -l2, -l3] + aux_absent,
            [-1, -l1, 0, -l3] + aux_absent,
            [-1, l1, 0, 0] + aux_false,
            [-1, 0, l2, 0] + aux_false,
            [-1, 0, 0, l3] + aux_false,
        ])
        clauses[:4] *= 1/math.sqrt(4*3)
        clauses[4:] *= 1/math.sqrt(4*4)

        S_prime[aux*10:(aux + 1)*10] = clauses

    return S_prime

def verify_sat_solution(inputs, outputs, solutions, clauses):
    success = True
    # pretty_print(sat_verify_inputs=inputs, sat_verify_outputs=outputs)
    pretty_print(clauses=clauses)

    # Use sets for cleaner lookup.
    solutions = [set(solution) for solution in solutions]
    inputs = [set(input_) | set([1]) for input_ in inputs]
    outputs = [set(output) for output in outputs]
    clauses = [set(clause) for clause in clauses]


    scores = []
    for solution in solutions:
        scores.append(0)
        for clause in clauses:
            scores[-1] += len(solution.intersection(clause))
    # pretty_print(solution_scores=list(zip([sorted(list(solution), key=abs) for solution in solutions], scores)))


    pretty_print(positive_check=None)
    best_solutions = []
    for input_, output in zip(inputs, outputs):
        best_solution = None
        best_score = float('inf') # -1
        for solution, score in zip(solutions, scores):
            if input_.issubset(solution) and score < best_score:
                best_score = score
                best_solution = solution
        
        if best_score == float('inf'): #-1:
            success = False
            print(f'No solution for {input_}, {output}')
        else:
            best_solutions.append(best_solution)
            print(f'{sorted(list(best_solution), key=abs)} solves {input_}, {output}, with a score of {best_score}.')

    pretty_print(negative_check=None)
    for input_, output in zip(inputs, outputs):
        valid_solutions = []
        for solution in best_solutions:
            if input_.issubset(solution):
                if output.issubset(solution):
                    valid_solutions.append(solution)
                else:
                    print(f'{sorted(list(solution), key=abs)} invalidates {input_, output}.')
                    success = False
                    break
        else:
            if valid_solutions:
                print(f'{[sorted(list(valid_solution), key=abs) for valid_solution in valid_solutions]} solve {input_, output}.')
            else:
                print(f'{input_, output} is not solved.')
                success = False

    return success


def extract_s_tilde(S, verbose=True, extract_weights=False, cuda=True):
    S_tilde_final = []
    weights = []
    errors = torch.zeros(1)
    if cuda: 
        errors = errors.cuda()

    for i in range(S.size()[1]):
        S_prime = S[:, i]

        min_error = None
        num_vars = None
        S_tilde_current = None
        C = None    
        for threshold in range(1, S.size()[0] + 1):
            
            topk = torch.topk(torch.abs(S_prime), threshold).indices
            signs = S_prime[topk].sign()

            S_tilde = torch.zeros(S_prime.size())
            if cuda: 
                S_tilde = S_tilde.cuda()

            S_tilde[topk] = signs
            if S_tilde[0] != -1 and torch.nonzero(S_tilde[1:]).size(0) > 0:
                if S_tilde[0] != 1:
                    threshold += 1

                S_tilde[0] = -1
                
            D_neg1 = 1/math.sqrt(4*threshold)
            S_ideal = S_tilde*D_neg1

            if extract_weights:
                for c in np.linspace(0, 10, 1000):
                    error = torch.norm(S_ideal*c - S_prime)

                    if min_error is None or min_error > error:
                        min_error = error
                        num_vars = threshold
                        S_tilde_current = S_tilde
                        C = round(c)
            else:
                error = torch.norm(S_ideal - S_prime)
                 
                if min_error is None or min_error > error:
                        min_error = error
                        num_vars = threshold
                        S_tilde_current = S_tilde

        S_tilde_final.append(S_tilde_current)
        weights.append(C)
        errors += min_error

        if verbose:
            print(f'Row {i} -- {S_tilde_current} at {min_error} and C = {C}')

    S_tilde = torch.stack(S_tilde_final)

    return S_tilde, weights, errors

def satnet_clauses_to_pysat(S_tilde):
    formatted_clauses = []
    for clause in S_tilde.tolist():
        formatted_clause = [int((i + 1)*element) for i, element in enumerate(clause) if element != 0]

        if formatted_clause: # Don't add zero clauses.
            formatted_clauses.append(formatted_clause)

    return formatted_clauses

@torch.no_grad()
def pysat_data_to_satnet(pysat_data, num_literals):
    result = torch.zeros(len(pysat_data), num_literals)
    for i, data in enumerate(pysat_data):
        data = list(data)
        positive_indices = list(filter(lambda x: x > 0, data))
        positive_indices = [x - 1 for x in positive_indices]
        result[i][positive_indices] = 1

    return result




def extract_clauses(model):

    S_tilde, weights, error = extract_s_tilde(model.S)
    pretty_print(S_tilde=S_tilde, S=model.S.t(), total_entries=int(torch.sum(torch.abs(S_tilde))), error=error)

    # MAXSAT - Solve S_tilde in order to assess whether SATNet learns correct clauses.
    formatted_clauses = satnet_clauses_to_pysat(S_tilde)

    # formatted_clauses = [
    #     [-1, 2, -3, -5],
    #     [-1, -2, 3, -6],
    #     [-1, 4, -5, -6],
    # ]

    formula = WCNF()
    formula.extend(formatted_clauses, weights=weights)
    pretty_print(weights_hard=formula.hard, weights_soft=formula.soft)

    solver = RC2(formula, verbose=0)

    solutions = [m for m in solver.enumerate()]
    pretty_print(solutions=solutions)

    # Check XOR.
    inputs = [
        [2, 3,],
        [2, -3,],
        [-2, 3,],
        [-2, -3,],
    ]

    outputs = [
        [-4],
        [4],
        [4],
        [-4],
    ]

    verify_sat_solution(inputs, outputs, solutions, formatted_clauses)

    # y = model(torch.FloatTensor([[1., 1., 0.]]).cuda(), torch.IntTensor([[1, 1, 0]]).cuda())
    # x = torch.FloatTensor([-1, 1, 1, 1, 1])
    # y_mine = S_tilde*x
    # pretty_print(model_y=y, clauses=y_mine) 
    return


@torch.no_grad()
def compute_error(pred, target):
    y = (pred-0.5)
    t = (target-0.5)
    correct = ((y * t).sign()+1.)/2
    acc = correct.sum().float()/target.size(0)

    return 1-float(acc)