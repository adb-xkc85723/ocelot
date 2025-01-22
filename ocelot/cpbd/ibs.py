"""
@author: Alex Brynes
Created on 22.01.2025
"""

from ocelot.common.globals import *
from ocelot.cpbd.elements import *
from ocelot.cpbd.coord_transform import *
from scipy.optimize import fmin
from scipy.special import expi
from ocelot.cpbd.physics_proc import PhysProc
from ocelot.cpbd.beam import global_slice_analysis, ParticleArray
from ocelot.cpbd.optics import lattice_transfer_map_z
from ocelot.cpbd.magnetic_lattice import MagneticLattice
import logging
import copy

logger = logging.getLogger(__name__)

class IBS(PhysProc):
    """
    Intrabeam Scattering (IBS) calculation physics process.
    Following `Nagaitsev`_., `Gjonaj`_.

    Attributes:
        self.lattice = `~ocelot.cpbd.magnetic_lattice.MagneticLattice` - lattice used during tracking
        self.step = 1 [in `~ocelot.cpbd.navi.Navigator.unit_step`] - step of the MBI calculation

    Description:
        The increase in slice energy spread caused by IBS is calculated sequentially along the beamline.
        At each simulation step, the bunch slice properties are calculated.
        Based on these parameters, the rms energy spread increase due to IBS is calculated.
        Each particle is given a kick in energy assuming a Gaussian distribution.

    .. _Nagaitsev: https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.8.064403
    .. _Gjonaj: https://indico.desy.de/event/47068/contributions/179045/attachments/94146/127982/DESY_EG_05.12.2024.pdf
    """
    def __init__(self, lattice: MagneticLattice, step: int=1):
        PhysProc.__init__(self, step)
        self.num_slice = 1
        self.lattice = lattice
        self.slice = None
        self.slice_params = {}

    def get_slice_params(self, p_array: ParticleArray):
        slice_params = global_slice_analysis(p_array)
        sli_cen = len(slice_params.s)/2
        sli_tot = len(slice_params.s)
        sli_min = int(sli_cen - (sli_tot/2))
        sli_max = int(sli_cen + (sli_tot/2))
        sli = {}
        for k, v in slice_params.__dict__.items():
            if k in ['beta_x', 'alpha_x', 'beta_y', 'alpha_y', 'I', 'sig_x', 'sig_y', 'sig_xp', 'sig_yp']:
                sli.update({k: np.mean(v[sli_min: sli_max])})
            elif k == 'se':
                sli.update({'sdelta': np.mean(v[sli_min: sli_max] / slice_params.me[sli_min: sli_max])})
            elif k == 'me':
                sli.update({'me': np.mean(v[sli_min: sli_max] / 1e9)})
                sli.update({'gamma': np.mean(v[sli_min: sli_max] / 1e6 / m_e_MeV)})
            elif k in ['ex', 'ey']:
                sli.update({k: np.mean(v[sli_min: sli_max])})
        for k in ['ex', 'ey']:
            sli[k] *= sli['gamma']
        sli.update({'s': p_array.s})
        sli.update({'q': np.sum(p_array.q_array)})
        sli.update({'bunch_length': np.std(p_array.tau())})
        return sli

    def apply(self, p_array, dz):
        """
        apply IBS kick

        :param p_array: `~ocelot.cpbd.beam.ParticleArray` object
        :param dz: step size for `~ocelot.cpbd.physics_proc.PhysProc`
        :return:
        """
        if dz < 1e-10:
            logger.debug(" IBS applied, dz < 1e-10, dz = " + str(dz))
            return
        logger.debug(" IBS applied, dz =" + str(dz))
        p_array_c = copy.deepcopy(p_array)
        self.slice_params = self.get_slice_params(p_array_c)
        if len(self.slice_params) > 1:
            sigd = abs(dz * self.sigd_ibs(self.slice_params))
            sigdvals = np.random.normal(0, sigd, len(p_array.rparticles[5]))
            p_array.rparticles[5] += [i for i in sigdvals]
            print(f"sdelta {self.slice_params['sdelta']} sigd {sigd}")

    def qmax(self, slice_params, distance):
        '''
        Eq. 22: Coulomb log function in terms of min/max scattering angle

        :param sliceparams: beam slice parameters
        :param distance: distance travelled

        :return: [log]
        '''
        numer = distance * (slice_params['q'] / q_e) * (ro_e**2)
        denom = 2 * (slice_params['ex'])**1.5 * slice_params['bunch_length'] * np.sqrt(slice_params['beta_x'])
        return np.sqrt(numer / denom)

    def coulomb_log(self, slice_params):
        return np.log(slice_params['gamma']**2 * slice_params['sig_xp']**2 * slice_params['sig_x'] / ro_e)

    def form_factor(self, slice_params):
        zeta = (slice_params['sdelta'] * slice_params['sig_x'] / slice_params['ex']) ** 2
        return (1 - zeta**0.25) * np.log(zeta + 1) / zeta

    def sigd_ibs(self, slice_params):
        logfac = self.coulomb_log(slice_params)
        npart = slice_params['q'] / q_e
        numer = (ro_e ** 2) * npart * logfac# * sdelta#  * logfac

        emit = np.sqrt(slice_params['ex'] * slice_params['ey'])
        sigx = slice_params['sig_x']
        bunchlength = slice_params['bunch_length']
        gamma = slice_params['gamma']
        speed_cub = np.sqrt((1 - (gamma ** -2))) ** 3

        # denom = 8 * speed_cub * emit * sigx * bunchlength * (gamma ** 2) * (sdelta ** 2)
        denom = 8 * speed_cub * emit * sigx * bunchlength# * (sdelta ** 2)

        formfac = self.form_factor(slice_params)

        return (numer / denom) * formfac