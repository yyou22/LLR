# Run training on OCT2017 dataset

run `main.py`

# Test robust accuracy on OCT2017 dataset

run `pgd_attack_oct.py`

# Run training on HAM10000 dataset

run `main_ham10000.py`

For existing experiment, we run RST (supervised version) with varying degrees of beta on the HAM10000 dataset

# Test robust accuracy on HAM10000 dataset

run `pgd_attack_ham10000.py`

# Run Simba

The folder SimBA has the scripts needed to conduct SimBA attack on the HAM10000 dataset (simple blackbox attack)

`run_simba_ham.py` conducts the attack. `simba_eva.py` evaluates the natural and adversairal accuracy under the simba attack.