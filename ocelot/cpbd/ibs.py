"""
@author: Igor Zagorodnov @ Martin Dohlus
Created on 27.03.2015
Revision on 01.06.2017: coordinate transform to the velocity direction
2019: Added LSC: S. Tomin and I. Zagorodnov
"""
from jedi.debug import speed

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

'''
Derived parameters relevant for calculating MBI gain. Mostly based on Sci. Rep. 11.7895
https://www.nature.com/articles/s41598-021-87041-0
'''

def sigd0(sigdE: float, E0: float) -> float:
    """
    initial fractional uncorrelated energy spread rms

    :param sigdE: absolute uncorrelated rms energy spread [GeV]
    :param E0: beam energy [GeV]
    :return: sigd_delta
    """
    return sigdE/E0

# R56 of DS1-chicane in [m]
def R56(theta, Lb, DL):
    return -2*(theta**2)*(((2/3)*Lb)+DL)


def D_bane(Q, bunchlength, enx):
    """
    constant factor in Bane's approximation

    :param Q: bunch charge [C]
    :param bunchlength: rms bunch length [m]
    :param enx: horizontal normalized emittance [m-rad]
    :return: D_bane
    """
    return (ro_e**2)*(Q/q_e)/(8*bunchlength*enx)

class IBS(PhysProc):
    """
    Intrabeam Scattering (IBS) calculation physics process

    Attributes:
        self.lattice = MagneticLattice - lattice used during tracking
        self.step = 1 [in Navigator.unit_step] - step of the MBI calculation

    Description:
        The increase in slice energy spread caused by IBS is calculated sequentially along the beamline.
        At each simulation step, the bunch slice properties and lattice transfer map up to that point are extracted.
        The bunching factor in the absence of collective effects -- b0 -- is also calculated.
        Based on these parameters, the microbunching integral kernel at each previous step is evaluated and multiplied with b0
        The bunching factor at a given location z is then the sum of all previous bunching factors
        These bunching factors [bf, pf] are then made attributes of the `~ocelot.cpbd.beam.ParticleArray` object.

    .. _Tsai et al: https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.23.124401
    """
    def __init__(self, lattice: MagneticLattice, step: int=1):
        PhysProc.__init__(self, step)
        self.smooth_param = 0.1
        self.step_profile = False
        self.napply = 0
        self.num_slice = 1
        self.lattice = lattice
        self.first = True
        #self.lattice = None
        self.zpos = 0
        self.slice = None
        self.dist = []
        self.slice_params = []
        self.optics_map = []

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
        wakes in V/pC

        :param p_array:
        :param dz:
        :return:
        """
        if dz < 1e-10:
            logger.debug(" LSC applied, dz < 1e-10, dz = " + str(dz))
            return
        logger.debug(" LSC applied, dz =" + str(dz))
        p_array_c = copy.deepcopy(p_array)
        self.slice_params.append(self.get_slice_params(p_array_c))
        self.z0 = self.slice_params[-1]['s'] - self.slice_params[0]['s']
        self.ltm, self.elem = lattice_transfer_map_z(self.lattice, self.slice_params[0]['me'], self.z0)
        self.optics_map.append(self.ltm)
        self.dist.append(self.z0)
        print('\n')
        if len(self.slice_params) > 1:
            distance = (self.slice_params[-1]['s'] - self.slice_params[-2]['s'])
            sigd = abs(distance * self.sigd_ibs(self.slice_params))
            print(sigd)
            sigdvals = np.random.normal(0, sigd, len(p_array.rparticles[5]))
            p_array.rparticles[5] += [i for i in sigdvals]
            print(f"sdelta {self.slice_params[-1]['sdelta']} sigd {sigd}")

    def qmax(self, slice_params, distance):
        '''
        Eq. 22: Coulomb log function in terms of min/max scattering angle

        :param sliceparams: beam slice parameters
        :param distance: distance travelled

        :return: [log]
        '''
        numer = distance * (slice_params[-1]['q'] / q_e) * (ro_e**2)
        denom = 2 * (slice_params[-1]['ex'])**1.5 * slice_params[-1]['bunch_length'] * np.sqrt(slice_params[-1]['beta_x'])
        return np.sqrt(numer / denom)

    def coulomb_log(self, slice_params):
        denom = ro_e / (slice_params[-1]['sig_xp'])**2
        # return np.log(slice_params[-1]['sig_x'] / denom)
        return np.log(slice_params[-1]['gamma']**2 * slice_params[-1]['sig_xp']**2 * slice_params[-1]['sig_x'] / ro_e)
        # return np.log(self.qmax(slice_params, distance) * np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['ey']) / (2 * np.sqrt(2) * ro_e))

    def gradient(self, slice_params, distance):
        return (slice_params[-1]['gamma'] - slice_params[-2]['gamma']) / (m_e_MeV * distance)

    def ibs_k(self, slice_params, distance):
        '''
        Eq. 28: factor for accelerating gradient

        :param slice_params: beam slice parameters
        :param distance: distance travelled

        :return: k
        '''
        grad = self.gradient(slice_params, distance)
        numer = ro_e * (slice_params[-1]['q'] / q_e) * m_e_MeV
        denom = 4 * grad * (slice_params[-1]['ex']**1.5) * (slice_params[-1]['beta_x']**0.5) * slice_params[-1]['bunch_length']
        return numer / denom

    def form_factor(self, slice_params):
        zeta = (slice_params[-1]['sdelta'] * slice_params[-1]['sig_x'] / slice_params[-1]['ex']) ** 2
        return (1 - zeta**0.25) * np.log(zeta + 1) / zeta

    def sigd_ibs(self, slice_params):
        logfac = self.coulomb_log(slice_params)
        npart = slice_params[-1]['q'] / q_e
        sdelta = slice_params[-1]['sdelta']
        numer = (ro_e ** 2) * npart * logfac# * sdelta#  * logfac

        emit = np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['ey'])
        sigx = slice_params[-1]['sig_x']
        bunchlength = slice_params[-1]['bunch_length']
        gamma = slice_params[-1]['gamma']
        speed_cub = np.sqrt((1 - (gamma ** -2))) ** 3

        # denom = 8 * speed_cub * emit * sigx * bunchlength * (gamma ** 2) * (sdelta ** 2)
        denom = 8 * speed_cub * emit * sigx * bunchlength# * (sdelta ** 2)

        formfac = self.form_factor(slice_params)

        return (numer / denom) * formfac

    # def sigd_ibs(self, slice_params, optics_map, distance, elem):
    #     '''
    #     Eq. 29: increase in sigma_delta
    #
    #     :param slice_params: beam slice parameters
    #     :param distance: distance travelled
    #
    #     :return: sigd
    #     '''
    #     if elem.__class__ == Cavity:
    #         # db = D_bane(slice_params[-1]['q'], slice_params[-1]['bunch_length'], slice_params[-1]['ex'])
    #         # banefactor = m_e_MeV * db / (self.gradient(slice_params, distance) * np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['beta_x']))
    #         # gammadifffactor = ((1 / np.sqrt(slice_params[-2]['gamma'])) - (1 / np.sqrt(slice_params[-1]['gamma'])))
    #         # # clog = self.coulomb_log(slice_params, distance)
    #         # clog = self.coulomb_log(slice_params, elem.l)
    #         # loggamfcub = np.log(slice_params[-1]['gamma'] ** 3)
    #         # gamfsqrt = np.sqrt(slice_params[-1]['gamma'])
    #         # loggamicub = np.log(slice_params[-2]['gamma'] ** 3)
    #         # gamisqrt = np.sqrt(slice_params[-2]['gamma'])
    #         # return np.sqrt(abs(banefactor * ((gammadifffactor * (4 * clog - 6)) + ((loggamfcub / gamfsqrt) - (loggamicub / gamisqrt)))))
    #         # db = D_bane(slice_params[-1]['q'], slice_params[-1]['bunch_length'], slice_params[-1]['ex'])
    #         # grad = self.gradient(slice_params, distance)
    #         # gam1 = slice_params[-1]['gamma']
    #         # gam0 = slice_params[-2]['gamma']
    #         # fac1 = (2 * m_e_MeV * db / (3 * (gam1**2) * grad * np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['beta_x'])))
    #         # fac21 = gam1**1.5 - gam0**1.5
    #         # fac22 = 2 * self.coulomb_log(slice_params, elem.l) * fac21
    #         # fac23 = (gam0**1.5 * np.log(gam0**0.75)) - (gam1**1.5 * np.log(gam1**0.75))
    #         # return np.sqrt(fac1 * (fac21 + fac22 + fac23))
    #         grad = abs(self.gradient(slice_params, distance))
    #         gam1 = slice_params[-1]['gamma']
    #         gam0 = slice_params[-2]['gamma']
    #         print(f'ibs_k \t {self.ibs_k(slice_params, distance)} gamfac \t {(gam1**1.5 - gam0**1.5)} denom \t {(grad / m_e_eV) * (gam1**2) * 3}')
    #         print(f'fin \t {np.sqrt(4 * self.ibs_k(slice_params, distance) * (gam1**1.5 - gam0**1.5) / ((grad / m_e_eV) * (gam1**2)))}')
    #         numer = 4 * self.ibs_k(slice_params, distance) * (gam1**1.5 - gam0**1.5)
    #         denom = (grad / m_e_eV) * (gam1**2)
    #         return np.sqrt(numer / denom)
    #     if self.dispersion_invariant_x(slice_params, optics_map) > 1e-6:
    #         afacn = (ro_e ** 2 * slice_params[-1]['q'] / q_e)
    #         afacd = 4 * (slice_params[-1]['gamma'] ** 2) * slice_params[-1]['ex'] * np.sqrt(
    #             slice_params[-1]['sig_x'] * slice_params[-1]['sig_y']) * slice_params[-1]['bunch_length']
    #         afac = afacn / afacd
    #         bfac = np.sqrt(distance * afac) * slice_params[-1]['ex'] / 2 / ro_e
    #         hfac = slice_params[-1]['gamma'] * self.dispersion_invariant_x(slice_params, optics_map) / slice_params[-1]['ex']
    #         sigdibs = (fmin(self.solBC, x0=slice_params[-1]['sdelta'], args=(slice_params[-2]['sdelta'], hfac, afac, bfac, distance)))
    #         return 0*np.sqrt(abs((sigdibs ** 2) - slice_params[-1]['sdelta'] ** 2))
    #     else:
    #         db = D_bane(slice_params[-1]['q'], slice_params[-1]['bunch_length'], slice_params[-1]['ex'])
    #         numer = 2 * db * distance * self.coulomb_log(slice_params, distance)
    #         denom = (slice_params[-1]['gamma'])**1.5 * np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['beta_x'])
    #         return np.sqrt((numer/denom))


    def dispersion_invariant_x(self, slice_params, optics_map):
        '''
        Eq. 8: H-functions

        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param index: index in lattice

        :return: Hx
        '''
        r16sq = optics_map[-1][0, 5] ** 2
        betax = slice_params[-1]['beta_x']
        alphax = slice_params[-1]['alpha_x']
        numerator = r16sq + ((betax * optics_map[-1][1, 5]) + (alphax * optics_map[-1][1, 5])) ** 2
        return numerator / betax

    def dispersion_invariant_y(self, slice_params, optics_map):
        '''
        Eq. 8: H-functions

        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param index: index in lattice

        :return: Hy
        '''
        r36sq = optics_map[-1][2, 5] ** 2
        betay = slice_params[-1]['beta_x']
        alphay = slice_params[-1]['alpha_x']
        numerator = r36sq + ((betay * optics_map[-1][4, 5]) + (alphay * optics_map[-1][3, 5])) ** 2
        return numerator / betay

    def solBC(self, y, sigd, hBC, aBC, bBC, distance):
        logfactor1 = 3 * np.log(np.sqrt(hBC * (sigd ** 2) + 1) / bBC)
        logfactor2 = 3 * np.log(np.sqrt(hBC * (y ** 2) + 1) / bBC)
        expfac1 = -expi(logfactor1)
        expfac2 = expi(logfactor2)
        return abs((2 * (bBC**3) / hBC) * (expfac1 + expfac2) - (aBC * distance))