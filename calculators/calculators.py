from ase.calculators.aims import Aims
import numpy as np
from tempfile import TemporaryDirectory
import signal
import logging

LOG = logging.getLogger(__name__)


def kpoints_per_klength(klength, kspacing=0.05):
    nk_raw = max(1, ((klength/kspacing)))
    return int(np.ceil(nk_raw))

def calc_kpoint_grid(atoms, kspacing=0.05):
    k_vec_norms = [
        np.linalg.norm(kvec) for kvec in atoms.cell.reciprocal() #length of each cell vector
    ] 
    nkvec = [kpoints_per_klength(knorm, kspacing) for knorm in k_vec_norms] #number of kpoints per vector length
    return nkvec



class AimsTotal():

    default_settings = {
        "xc": "pbe",
        "sc_accuracy_eev": 10e-3,
        "sc_accuracy_etot": 10e-6,
        "many_body_dispersion": "",
        "compute_forces": True,
    }


    def __init__(self, ref_ncores, **kwargs):
        self.ncores = ref_ncores
        self.system_settings = self.get_system_settings()
        self.calculation_settings = self.__class__.default_settings.copy()
        self.calculation_settings.update(kwargs)


    def get_system_settings(self):
        raise NotImplementedError("system settings for calculator needed")
        # return {
            "command": f"mpiexec -np {self.ncores} /path/to/aims_binary.mpi" > aims.out",
            "species_dir": "/path/to/species/basis_set",
            "tier": 2,
        }

    def get_calculator(self, working_dir='.', kpoints=None,):
        return Aims(
            kpts=kpoints,
            directory=working_dir,
            **self.system_settings,
            **self.calculation_settings,

        )
    
    def calculate(self, atoms):
        timeout = 1800
        def handler(signum, frame):
            raise TimeoutError("calculation failed") 

        kpoints = calc_kpoint_grid(atoms)

        # set up calculator
        with TemporaryDirectory(dir=".") as tmpdir:
            atoms.calc = self.get_calculator(tmpdir, kpoints=kpoints)
            # get energy and forces, error handling
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
            except Exception as e:
                LOG.error(e)
                energy, forces = (None, None)
            finally:
                signal.alarm(0)
            
        return energy, forces
