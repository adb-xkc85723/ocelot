"""
@author: Alex Brynes
Created on 22.01.2025
"""

from ocelot.common.globals import ro_e, q_e
from ocelot.cpbd.coord_transform import *
from ocelot.cpbd.physics_proc import PhysProc
from ocelot.cpbd.beam import get_envelope
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
    def __init__(self, step: int=1):
        PhysProc.__init__(self, step)
        self.num_slice = 1
        self.slice_params = {}
        self.slice = "Imax"
        self.emit_n = None
        self.sigma_xy = None
        self.sigma_z = None
        self.sigma_dgamma0 = None
        self.gamma = None
        self.sigma_xpyp = None
        self.q = None

    def get_beam_params(self, p_array):
        """
        Compute beam parameters such as sigma_xy, sigma_z, and normalized emittance.

        :param p_array: ParticleArray
            Input particle array containing particle properties.
        """
        tws = get_envelope(p_array)#, bounds=self.bounds, slice=self.slice)
        self.sigma_z = np.std(p_array.tau())
        self.sigma_xy = (np.sqrt(tws.xx) + np.sqrt(tws.yy)) / 2.
        self.sigma_xpyp = (np.sqrt(tws.pxpx) + np.sqrt(tws.pypy)) / 2.
        self.emit_n = (tws.emit_xn + tws.emit_yn) / 2.
        pc = np.sqrt(p_array.E ** 2 - m_e_GeV ** 2)
        self.sigma_dgamma0 = np.sqrt(tws.pp)*pc / m_e_GeV
        self.gamma = p_array.E / m_e_GeV
        self.q = np.sum(p_array.q_array)

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
        self.get_beam_params(p_array_c)
        sigd = abs(dz * self.sigd_ibs())
        sigdvals = np.random.normal(0, sigd, len(p_array.rparticles[5]))
        p_array.rparticles[5] += [i for i in sigdvals]
        print(f"sdelta {self.sigma_dgamma0} sigd {sigd}")

    def coulomb_log(self):
        return np.log(self.gamma**2 * self.sigma_xpyp**2 * self.sigma_xy / ro_e)

    def form_factor(self):
        zeta = (self.sigma_dgamma0 * self.sigma_xy / self.emit_n) ** 2
        return (1 - zeta**0.25) * np.log(zeta + 1) / zeta

    def sigd_ibs(self):
        logfac = self.coulomb_log()
        npart = self.q / q_e
        numer = (ro_e ** 2) * npart * logfac# * sdelta#  * logfac

        speed_cub = np.sqrt((1 - (self.gamma ** -2))) ** 3
        # denom = 8 * speed_cub * emit * sigx * bunchlength * (gamma ** 2) * (sdelta ** 2)
        denom = 8 * speed_cub * self.emit_n * self.sigma_z# * (sdelta ** 2)

        formfac = self.form_factor()

        return (numer / denom) * formfac