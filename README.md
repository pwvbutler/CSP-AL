# CSP-AL

Implementation of active learning strategies, primarily intended for use with crystal structure prediction data. Note the code here was designed to be model agnostic and as such there is no implementation/interface for the machine learned potential to be trained. The current query strategies available are: random sampling, highest uncertainty sampling, highest uncertainty FPS. 

An example reference calculator using the ASE interface to FHI-aims is provided.

Requirements:
  - python>=3.8
  - ASE
  - numpy

# Usage
The main script is `mlp_trainer.py`. Invoke the active learning with an extended xyz containing the candidate structures by:
```
python mlp_trainer.py /path/to/candidates.xyz <reference-calculator> -m <query-strategy>
```

Options for the active learning are specified by command line args:
```
usage: mlp_trainer.py [-h] -m {random,highest_uncertainty,highest_uncertainty_FPS}
                      [-ce RIGID_CONFORMATIONAL_ENERGY] [--delta-learning]
                      [--energy-cutoff ENERGY_CUTOFF] [--uncertainty-threshold UNCERTAINTY_THRESHOLD]
                      [--uncertain-target UNCERTAIN_TARGET] [--max-dataset-size MAX_DATASET_SIZE]
                      [--batchsize BATCHSIZE] [-j NCORES] [-rs RANDOM_SEED]
                      [--log-level {INFO,DEBUG,WARNING,ERROR}]
                      candidates_xyz reference_method

positional arguments:
  candidates_xyz        extxyz containing candidate structures
  reference_method      calculator for reference data

optional arguments:
  -h, --help            show this help message and exit
  -m {random,highest_uncertainty,highest_uncertainty_FPS}, --query-strategy {random,highest_uncertainty,highest_uncertainty_FPS}
                        active learning method/query strategy
  -ce RIGID_CONFORMATIONAL_ENERGY, --rigid-conformational-energy RIGID_CONFORMATIONAL_ENERGY
                        conformational energy to be subtracted from reference total energies
  --delta-learning      train NNP by delta learning, i.e. diff between CSP energy and ref method
  --energy-cutoff ENERGY_CUTOFF
                        only train on structures with energy less than cutoff
  --uncertainty-threshold UNCERTAINTY_THRESHOLD
                        threshold MLP uncertainty to be considered poorly described
  --uncertain-target UNCERTAIN_TARGET
                        target max percentage of uncertain candidates (for highest uncertainty
                        strategy)
  --max-dataset-size MAX_DATASET_SIZE
                        max training structures selected
  --batchsize BATCHSIZE
                        number of structures selected before retraining MLP
  -j NCORES, --ncores NCORES
                        number of cores for parallel processes
  -rs RANDOM_SEED, --random-seed RANDOM_SEED
                        seed to set rng
  --log-level {INFO,DEBUG,WARNING,ERROR}
                        Log level
```

Delta learning is implemented using attributes from the extxyz. If the conformation is the same across structures (e.g. rigid CSP), it is possible to provide a conformational energy and thus calculate intermolecular energies which can then be used to delta learn with intermolecular force fields. 
