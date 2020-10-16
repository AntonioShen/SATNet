import pytest
import math
import random
from satnet.utilities import pysat_data_to_satnet, compute_error, extract_s_tilde, satnet_clauses_to_pysat, verify_sat_solution
from satnet import SATNet

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

from tqdm.auto import tqdm 

def generate_random_k_cnf(num_literals, num_clauses, p_literal_positive=0.5, p_literal_included=0.2):
    cnf = []

    for clause in range(num_clauses):
        clause = []
        for literal in range(num_literals):
            if random.random() < p_literal_included:
                value = -1*(1 + literal) if random.random() < p_literal_positive else 1 + literal
                clause.append(value)

        if clause:
            cnf.append(clause)

    return cnf

def generate_and_cnf(num_literals, num_clauses, p_literal_positive=0.5, p_literal_included=0.2):
    return [
            {-1, -2, 3},
            {1, -3},
            {2, -3},
        ]

def generate_or_cnf(num_literals, num_clauses, p_literal_positive=0.5, p_literal_included=0.2):
    return [
            {1, 2, -3},
            {-1, 3},
            {-2, 3},
        ]

def generate_xor_cnf(num_literals, num_clauses, p_literal_positive=0.5, p_literal_included=0.2):
    return [
            {-1, -2, -3},
            {1, 2, -3},
            {1, -2, 3},
            {-1, 2, 3},
        ]

def generate_dataset(clause_generation_function, num_inputs, num_literals, num_clauses, p_literal_positive=0.5, p_literal_included=0.2):
    while True:
        cnf = clause_generation_function(num_literals, num_clauses, p_literal_positive, p_literal_included)

        formula = WCNF()
        formula.extend(cnf)

        solver = RC2(formula, verbose=0)
        solutions = [m for m in solver.enumerate()]

        multiple_outputs_for_one_input = False
        for i, solution in enumerate(solutions):
            for j in range(i):
                if solution[:num_inputs] == solutions[j][:num_inputs]:
                    multiple_outputs_for_one_input = True

        if solutions and not multiple_outputs_for_one_input:
            break

    solutions = [set(solution) for solution in solutions]
    dataset = []

    def recursive_for(solutions, num_inputs):

        def recursive_for_impl(solutions, previous_set_inputs, current_input, num_inputs):
            if num_inputs < current_input:
                result = []
                for solution in solutions:
                    if previous_set_inputs.issubset(solution):
                        result.append(solution)
                return result

            return (
                recursive_for_impl(solutions, previous_set_inputs | set([current_input]), current_input + 1, num_inputs) 
                + recursive_for_impl(solutions, previous_set_inputs | set([-current_input]), current_input + 1, num_inputs)
            )

        assert num_inputs > 1

        return recursive_for_impl(solutions, set(), 1, num_inputs)
        
    valid_data = recursive_for(solutions, num_inputs)

    dataset = pysat_data_to_satnet(valid_data, num_literals)

    return solutions, dataset

def sign(x):
    return -1 if x < 0 else 1

def check_clauses(model, expected_solutions, num_inputs, num_outputs):
    S_tilde, weights, error = extract_s_tilde(model.S, verbose=False, cuda=False)
    print('S_TILDE', S_tilde)

    formatted_clauses = satnet_clauses_to_pysat(S_tilde)

    formula = WCNF()
    formula.extend(formatted_clauses, weights=weights)
    solver = RC2(formula, verbose=0)

    solutions = [m for m in solver.enumerate()]

    inputs = []
    outputs = []

    for expected_solution in expected_solutions:
        expected_solution = sorted(list(expected_solution), key=lambda x : abs(x))
        expected_solution = [(abs(x) + 1)*sign(x) for x in expected_solution]
        inputs.append(expected_solution[:num_inputs])
        outputs.append(expected_solution[num_inputs:num_inputs + num_outputs])

    return verify_sat_solution(inputs, outputs, solutions, formatted_clauses)


def run_test(generation_function, batch_size = 128, num_epochs = 10, beta = 0.1, num_inputs = 2, num_outputs = 1, num_aux = 7, num_clauses = 8):
    NUM_LITERALS = num_inputs + num_outputs + num_aux

    min_aux = 2**math.ceil(math.log(num_inputs + num_outputs, 2))
    model = SATNet(num_inputs + num_outputs, num_clauses, min_aux, prox_lam=1e-1, eps=1e-4, max_iter=100)
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    solutions, dataset = generate_dataset(generation_function, num_inputs, NUM_LITERALS, num_clauses)

    input_mask = torch.IntTensor([[1]*num_inputs + [0]*num_outputs]).repeat(dataset.size(0), 1)

    dataset = TensorDataset(dataset, input_mask)

    for i in range(num_epochs):
        loader = DataLoader(dataset, batch_size=batch_size)
        tloader = tqdm(enumerate(loader), total=len(loader))
        
        for j, (data, mask) in tloader:
            optimizer.zero_grad()

            y = model(data[:, :num_inputs + num_outputs], mask)
            prediction = y[:,-num_outputs:]
            label = data[:,num_inputs:num_inputs+num_outputs]

            _, _, s_tilde_error = extract_s_tilde(model.S, verbose=False, cuda=False)
            loss = F.binary_cross_entropy(prediction, label) + beta*s_tilde_error
            error = compute_error(prediction, label)

            loss.backward()
            optimizer.step()

            tloader.set_description(f'Epoch {i}: loss {float(loss):.3f}, accuracy {1 - float(error):.3f}, S~ error {float(s_tilde_error):.3f}')

    # print('------------------------------ddddddddddddddddddddddddddddddd', model.S, 1000*torch.randn_like(model.S))
    # model.S = torch.nn.Parameter(model.S + 1000*torch.randn_like(model.S))
    S = torch.FloatTensor([
            [-1, 1, -1, 0, 0, -1],
            [-1, -1, 1, 0, -1, 0],
            [-1, 0, 0, 1, -1, -1],
            [0, 0, 0, 0, 0, 0],
        ])

    # S = torch.FloatTensor([
    #         [-1, 1, 1, -1],
    #         [-1, 1, -1, 1],
    #         [-1, -1, 1, 1],
    #         [-1, -1, -1, 1],
    #     ])    
    model.S = torch.nn.Parameter(S.t())

    assert check_clauses(model, solutions, num_inputs, num_outputs)

    
def test_and():
    run_test(generate_xor_cnf, beta=0.2, num_epochs=1)

# def test_or():
#     run_test(generate_and_cnf)

# def test_xor():
#     run_test(generate_and_cnf)

# @pytest.mark.parametrize('repeat', [i for i in range(10)])
# def test_random(repeat):
#     run_test(generate_random_k_cnf, num_inputs=3, num_outputs=2, num_aux=0, beta=0)