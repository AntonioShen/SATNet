## Expected Learned Clauses For Parity Dataset

| __a__ | __b__ | __x__ | __x = a XOR b__ |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 1 | 
| 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 | 
| 0 | 1 | 1 | 1 | 
| 1 | 0 | 0 | 0 | 
| 1 | 0 | 1 | 1 | 
| 1 | 1 | 0 | 1 | 
| 1 | 1 | 1 | 0 | 


## CNF
(a ∨ b ∨ ¬x) ∧ (a ∨ ¬b ∨ x) ∧ (¬a ∨ b ∨ x) ∧ (¬a ∨ ¬b ∨ ¬x)

## DNF
(¬a ∧ ¬b ∧ ¬x) ∨ (a ∧ ¬b ∧ x) ∨ (¬a ∧ b ∧ x) ∨ (a ∧ b ∧ ¬x)


## S~ Matrix

|  |  |  | |
|:---:|:---:|:---:|:---:|
| 0 | 1 | 1 | -1 | 
| 0 | 1 | -1 | 1 |
| 0 | -1 | 1 | 1 | 
| 0 | -1 | -1 | -1 | 


# Invocation

    python exps/parity.py --batchSz 1 --testBatchSz 1 --m 8 --aux 1 --model logs/parity.aux1-m8-lr0.1-bsz100/it2.pth --extract-clauses


