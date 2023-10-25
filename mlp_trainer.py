import logging
import random
import numpy as np
from pathlib import Path

from utils import generate_datafile_str, parse_datafile_content


LOG = logging.getLogger(__name__)


def get_MLP_parameters():
    LOG.warning("MLP not implemented. Will not train MLP")
    return None


class MLPTrainer():
    """
    Class for training neural network potentials on the fly from csp dbs
    """
    
    def __init__(
            self,
            candidates_xyz,
            query_strategy,
            reference_method,
            train_interval,
            max_dataset_size,
            ncores,
            **kwargs
        ):

        query_strategies = {
            "random": self.select_candidates_random,
            "highest_uncertainty": self.select_candidates_highest_uncertainty,
            "highest_uncertainty_FPS": self.select_candidates_highest_uncertainty_FPS,
        }

        # active learning
        self.finished = False
        self.candidates_xyz = candidates_xyz
        self.train_interval = train_interval
        self.uncertainty_threshold = kwargs.get("uncertainty_threshold")
        self.max_dataset_size = max_dataset_size
        self.delta_learning = kwargs.get("delta_learning", False)
        self.uncertain_percent_goal = kwargs.get("uncertain_percent_goal")
        self.query_strategy = query_strategy
        self.select_candidates = query_strategies[query_strategy]
        self.candidates = {}

        # reference calculations
        self.ref_ncores = kwargs.get("ref_ncores", ncores)
        self.reference_method = reference_method
        self.reference_calculator = self.get_reference_calculator(
            reference_method, ref_ncores=self.ref_ncores
        )
        self.conformational_energy = kwargs.get("conformational_energy", None)
        self.reference_dataset = {}
        self.intermol_data = {}

        # MLP settings
        self.trained_MLP = False
        self.MLP_parameters = get_MLP_parameters()

        # general settings
        self.ncores = ncores
        
        # setup rng
        random.seed(kwargs.get("random_seed", None))

        # restart active learning if input.data file exists in working dir
        if Path('input.data').exists():
            self.restart_from_input_data()



    def get_reference_calculator(self, reference_method, **kwargs):
        """initialize the specified reference calculator

        Parameters
        ----------
        reference_method : str
            the reference calculator name

        """
        from calculators import implemented_calculators

        if reference_method not in implemented_calculators:
            raise NotImplementedError(
                f"{reference_method} calculator not implemented.\n"
                f"implemented calculators: {list(implemented_calculators.keys())}"
            )
        
        return implemented_calculators[reference_method](**kwargs)

    def write_training_data(self):
        """convert training structures to input.data file"""

        content = ""
        for id, data in self.reference_dataset.items():
            content += generate_datafile_str(
                id,
                data['lattice_vectors'],
                data['atom_records'],
                data['energy'],
                data['charge'],
            )
        
        with open('input.data', 'w') as f:
            f.write(content)

    def restart_from_input_data(self):
        """
        restart active learning from the current training set stored as an input.data file
        """
        structure_data = parse_datafile_content(Path('input.data').read_text())
        
        for structure in structure_data:
            comment, lattice_vectors, atom_records, energy, charge = structure
            id = comment.split()[-1]
            
            atoms = [ (x['element'], *x['position'], *x['forces']) for x in atom_records ]
            
            self.reference_dataset[id] = {
                'energy': energy,
                'charge': charge,
                'lattice_vectors': lattice_vectors,
                'atom_records' : atoms,
            }
        
        self.train_model()
        self.trained_MLP = True

    
    def load_candidates(self):
        """
        read in candidates from extended xyz file
        """
        from ase import io
        
        candidates = [
            (i, candidate) for i, candidate in 
            enumerate(io.read(self.candidates_xyz, format="extxyz", index=':'))
        ]

        # shuffle candidates to random ordering
        random.shuffle(candidates)

        for (index, candidate) in candidates:
            
            name = candidate.info.get("title", index)
            
            if name not in self.candidates:
                self.candidates[name] = CandidateStructure(
                    name=name,
                    index=index,
                    csp_energy=candidate.info.get("energy", None),
                    atoms=candidate,
                )

    def calculate_candidate_ref_data(self, candidate):
        """calculate the reference data for a given candidate

        Parameters
        ----------
        candidate : ase.Atoms
            the candidate to be calculated

        Returns
        -------
        _type_
            _description_
        """
        from calculators import AimsTotal

        # calculate ref data
        #calc = AimsTotal(ncores=self.ncores)
        calc = self.reference_calculator


        try:
            ref_energy, ref_forces = calc.calculate(candidate.atoms)
        except:
            LOG.warning("ref calculation failed for: %s", candidate.name)
            return None

        #if conformational energy, subtract
        if self.conformational_energy is not None:
            ref_energy -= self.conformational_energy
        
        # if delta learning, calc deltas
        if self.delta_learning:
            ref_energy -= candidate.csp_energy
            ref_forces -= candidate.atomic_forces

        return ref_energy, ref_forces

    def add_ref_data_to_training_set(self, candidate, ref_energy, ref_forces, **kwargs):
        """add calculated data for a candidate to the training set

        Parameters
        ----------
        candidate : ase.Atoms
            _description_
        ref_energy : float
            calculated energy
        ref_forces : [float]
            calculated forces, in same order as positions
        """
        charge = candidate.atoms.info.get("charge", 0.0)
        elements = candidate.atoms.get_chemical_formula(mode="all")
        positions = candidate.atoms.get_positions()
        lattice_vectors = candidate.atoms.get_cell()
        atom_records = [ # (element, x, y, z, fx, fy, fz)
            (x[0], *x[1], *x[2]) for x in zip(elements, positions, ref_forces) 
        ]
        self.reference_dataset[candidate.name] = {
            'energy': ref_energy,
            'charge': charge,
            'lattice_vectors': lattice_vectors,
            'atom_records' : atom_records,
        }

        self.write_training_data()

    def train_model(self):
        """
        train the MLP with the current training set.
        """
        pass
            
    def calculate_candidate_descriptors(self):
        """
        calculate descriptors for candidates in the candidate list
        """
        pass
        
    def calculate_distance_matrix(self, candidates):
        """
        calculate distance matrix between structures for FPS sampling
        
        candidates <list>: candidate names to construct distance matrix with
        """
        pass

    def evaluate_candidates_uncertainty(self, candidates):
        """evaluate candidate using MLP and return uncertainty"""
        pass

    def get_high_uncertainty_candidates(self):
        """
        evaluate candidates and return candidates above uncertainty threshold
        """
        num_candidates = len(self.candidates)
        LOG.info(f"evaluating %d candidate structures", num_candidates)
        uncertainties = self.evaluate_candidates_uncertainty(self.candidates)
        
        for i, name in enumerate(self.candidates):
            self.candidates[name].uncertainty = uncertainties[i]
        
        LOG.info(
            "max uncertainty in candidates %.4f; average uncertainty %.4f",
            max(uncertainties), sum(uncertainties)/num_candidates,
        )
        
        # check percent of candidates above uncertainty threshold
        num_uncertain = len([x for x in uncertainties if x > self.uncertainty_threshold])
        ratio = num_uncertain / num_candidates
        LOG.info("ratio of candidates above uncertainty threshold: %.4f (%d/%d)", 
            ratio,
            num_uncertain,
            num_candidates,
        )
        if ratio < self.uncertain_percent_goal:
            self.finished = True
            return
        
        # sort by uncertainty
        candidates_iter = iter(sorted(
            self.candidates,
            key=lambda x: self.candidates[x].uncertainty,
            reverse=True
        ))

        return [
            name for name in candidates_iter
            if self.candidates[name].uncertainty > self.uncertainty_threshold
        ]
    
    def select_candidates_random(self):
        """
        select candidates sequentially from candidate list. This is equivalent 
        to random sampling since the candidate list is randomly ordered.
        """
        candidates_iter = iter(self.candidates.copy())
        num_calculated = 0

        while self.candidates and num_calculated < self.train_interval:
            name = next(candidates_iter)
            candidate = self.candidates.pop(name)

            # check if structure poorly described by model
            if self.trained_MLP:
                uncertainty = self.evaluate_candidate_uncertainty(candidate)
                
                if uncertainty is not None and uncertainty < self.uncertainty_threshold:
                    continue

            (ref_energy, ref_forces) = self.calculate_candidate_ref_data(candidate)
            
            if ref_energy is None:
                continue

            self.add_ref_data_to_training_set(candidate, ref_energy, ref_forces)

            num_calculated += 1
        
        if num_calculated == 0:
            self.finished = True

    def select_candidates_highest_uncertainty(self):
        """
        select candidates in order of highest uncertainty as evaluated by MLP
        """        
        
        # evaluate candidates
        if self.trained_MLP:
            candidates_iter = iter(self.get_high_uncertainty_candidates())
        else:
            candidates_iter = iter(self.candidates.copy())
        
        num_calculated = 0
        
        # calculate ref data upto training interval
        while self.candidates and num_calculated < self.train_interval:
            name = next(candidates_iter)
            candidate = self.candidates.pop(name)
            
            (ref_energy, ref_forces) = self.calculate_candidate_ref_data(candidate)
            
            if ref_energy is None:
                continue

            self.add_ref_data_to_training_set(candidate, ref_energy, ref_forces)
        
        if num_calculated == 0:
            self.finished = True
    

    def select_candidates_highest_uncertainty_FPS(self):
        """
        select candidates by farthest point sampling candidates above the uncertainty 
        threshold starting from the candidate with the highest uncertainty.
        """
        
        # evaluate candidates
        if self.trained_MLP:
            uncertain_candidates = self.get_high_uncertainty_candidates()
        else:
            # if not trained, do FPS on entire candidate set
            uncertain_candidates = list(self.candidates)
        
        num_calculated = 0
        LOG.info("calculating distance matrix between high uncertainty candidates")
        distance_matrix = self.calculate_distance_matrix(uncertain_candidates)
        uncertain_candidates = {i: name for i, name in enumerate(uncertain_candidates)}
        
        # calculate ref data upto training interval
        while uncertain_candidates and num_calculated < self.train_interval:
            # begin FPS from highest uncertainty structure
            next_highest_ind = list(uncertain_candidates.keys())[0]
            next_highest_name = uncertain_candidates[next_highest_ind]
            if num_calculated == 0 and 0 in uncertain_candidates:
                current_idx = 0
                shortest_distances = distance_matrix[current_idx]
            elif self.trained_weights and self.candidates[next_highest_name].uncertainty > 10.0:   
                current_idx = next_highest_ind
                LOG.info("adding very high uncertainty candidate (uncertainty = %s)",
                    self.candidates[uncertain_candidates[current_idx]].uncertainty
                )
                shortest_distances = np.minimum(
                    shortest_distances,
                    distance_matrix[current_idx],
                )
            else:
                current_idx = np.argmax(shortest_distances)
                shortest_distances = np.minimum(
                    shortest_distances,
                    distance_matrix[current_idx],
                )
                
            LOG.debug(f"{current_idx=}")
            LOG.debug(f"{uncertain_candidates=}")
            LOG.debug(f"{shortest_distances=}")
                
            name = uncertain_candidates.pop(current_idx)
            candidate = self.candidates.pop(name)
            
            (ref_energy, ref_forces) = self.calculate_candidate_ref_data(candidate)
            
            if ref_energy is None:
                continue

            self.add_ref_data_to_training_set(candidate, ref_energy, ref_forces)
        
        if num_calculated == 0:
            self.finished = True
        

    def run(self):
        """
        Run active learning on candidate list. Retrains MLP when the training set has 
        increased by the specified batch size.
        """
        
        self.load_candidates()
        LOG.info("loaded %d candidates", len(self.candidates))
        if self.query_strategy == "highest_uncertainty_FPS":
            self.calculate_candidate_descriptors()
        num_datapoints = 0
        
        while len(self.candidates) > 0 and num_datapoints < self.max_dataset_size:
            self.select_candidates()
            if self.finished:
                LOG.info("No furher datapoints to train on, finishing run.")
                break

            num_datapoints = len(self.reference_dataset)
            
            LOG.info("training network with %d datapoints", num_datapoints)
            self.train_model()
            


class CandidateStructure():
    """
    class for cadidate structure data for training MLP from CSP landscapes
    """
    
    def __init__(self, name, index, atoms, csp_energy, **kwargs):
        self.name = name
        self.index = index
        self.atoms = atoms
        self.csp_energy = csp_energy
        self.csp_forces = kwargs.get("csp_forces", None)
        self.descriptor = kwargs.get("descriptor", None)

    @property
    def atom_forces(self):

        if self.csp_forces:
            return self.csp_forces

        try:
            return self.atoms.get_forces()
        except RuntimeError as exc:
            LOG.error("unable to get forces for {}".format(self.name))
        


def main():
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "candidates_xyz",
        type=str,
        help='extxyz containing candidate structures',
    )

    parser.add_argument(
        "reference_method",
        type=str,
        help="calculator for reference data",
    )
    
    
    # ACTIVE LEARNING OPTIONS
    parser.add_argument(
        "-m",
        "--query-strategy",
        required=True,
        choices=["random", "highest_uncertainty", "highest_uncertainty_FPS"],
        help="active learning method/query strategy",
    )

    parser.add_argument(
        "-ce",
        "--rigid-conformational-energy",
        type=float,
        default=None,
        help="conformational energy to be subtracted from reference total energies",
    )
    
    parser.add_argument(
        "--delta-learning",
        action="store_true",
        default=False,
        help="train NNP by delta learning, i.e. diff between CSP energy and ref method",
    )
    
    parser.add_argument(
        "--energy-cutoff",
        type=float,
        default=None,
        help="only train on structures with energy less than cutoff",
    )
    
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=1.0,
        help="threshold MLP uncertainty to be considered poorly described",
    )
    
    parser.add_argument(
        "--uncertain-target",
        type=float,
        default=0.05,
        help="target max percentage of uncertain candidates (for highest uncertainty strategy)",
    )
    
    parser.add_argument(
        "--max-dataset-size",
        type=int,
        default=2000,
        help="max training structures selected",
    )
    
    parser.add_argument(
        "--batchsize",
        type=int,
        default=30,
        help="number of structures selected before retraining MLP",
    )
    
    
    # MISC OPTIONS
    parser.add_argument(
        "-j",
        "--ncores",
        default=4,
        type=int,
        help="number of cores for parallel processes",
    )
    
    parser.add_argument(
        "-rs",
        "--random-seed",
        type=int,
        default=None,
        help="seed to set rng",
    )
    
    parser.add_argument(
        "--log-level",
        choices=["INFO", "DEBUG", "WARNING", "ERROR",],
        default="INFO", 
        help="Log level"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    kwargs = vars(args)
    candidates_xyz = Path(kwargs.pop("candidates_xyz")).absolute()
    reference_method = kwargs.pop("reference_method")
    query_strategy = kwargs.pop("query_strategy")
    train_interval = kwargs.pop('batchsize')
    max_dataset_size = kwargs.pop("max_dataset_size", 2000)
    uncertainty_threshold = kwargs.pop("uncertainty_threshold")
    uncertain_percent_goal =  kwargs.pop("uncertain_target")
    
    
    trainer = MLPTrainer(
        candidates_xyz=candidates_xyz,
        reference_method=reference_method,
        query_strategy=query_strategy,
        train_interval=train_interval,
        max_dataset_size=max_dataset_size,
        uncertainty_threshold=uncertainty_threshold,
        uncertain_percent_goal=uncertain_percent_goal,
        **kwargs
    )

    trainer.run()

if __name__ == "__main__":
    main()
