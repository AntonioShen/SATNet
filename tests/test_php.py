import pytest
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

def generate_random_k_cnf_dataset(num_inputs, num_literals, num_clauses, p_literal_positive=0.5, p_literal_included=0.2):
    while True:
        random_kcnf = generate_random_k_cnf(num_literals, num_clauses, p_literal_positive, p_literal_included)
        random_kcnf = [
            {-1, -2, 3},
            {1, -3},
            {2, -3},
        ]

        formula = WCNF()
        formula.extend(random_kcnf)

        solver = RC2(formula, verbose=0)
        solutions = [m for m in solver.enumerate()]
        if solutions:
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


def test_parity():
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    BETA = 0.1

    NUM_INPUTS = 2
    NUM_OUTPUTS = 1
    NUM_AUX = 7
    NUM_CLAUSES = 8

    NUM_LITERALS = NUM_INPUTS + NUM_OUTPUTS + NUM_AUX

    model = SATNet(NUM_INPUTS + NUM_OUTPUTS, NUM_CLAUSES, NUM_AUX, prox_lam=1e-1, eps=1e-4, max_iter=100)
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    solutions, dataset = generate_random_k_cnf_dataset(NUM_INPUTS, NUM_LITERALS, NUM_CLAUSES)

    input_mask = torch.IntTensor([[1]*NUM_INPUTS + [0]*NUM_OUTPUTS]).repeat(dataset.size(0), 1)

    dataset = TensorDataset(dataset, input_mask)

    for i in range(NUM_EPOCHS):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        tloader = tqdm(enumerate(loader), total=len(loader))
        
        for j, (data, mask) in tloader:
            optimizer.zero_grad()

            y = model(data[:, :NUM_INPUTS + NUM_OUTPUTS], mask)
            prediction = y[:,-NUM_OUTPUTS].unsqueeze(1)
            label = data[:,NUM_INPUTS:NUM_INPUTS+NUM_OUTPUTS]

            _, _, s_tilde_error = extract_s_tilde(model.S, verbose=False, cuda=False)
            loss = F.binary_cross_entropy(prediction, label) + BETA*s_tilde_error
            error = compute_error(prediction, label)

            loss.backward()
            optimizer.step()

            tloader.set_description(f'Epoch {i}: loss {loss}, accuracy {error}')

    assert check_clauses(model, solutions, NUM_INPUTS, NUM_OUTPUTS)

    