from __future__ import division, print_function
import numpy as np
from openmdao.api import ExplicitComponent

class BregRange(ExplicitComponent):
    """
    Computes the fuel burn using the Breguet range equation using
    the computed CL, CD, weight, and provided specific fuel consumption, speed of sound,
    Mach number, initial weight, and range.

    Note that we add information from each lifting surface.

    Parameters
    ----------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    CT : float
        Specific fuel consumption for the entire aircraft.
    sonic_speed : float
        The Mach speed, speed of sound, at the specified flight condition.
    R : float
        The total range of the aircraft, used to backcalculate the fuel mass.
    Mach_number : float
        The Mach number of the aircraft at the specified flight condition.
    emptyTotal : float
        The operating empty weight of the aircraft. Supplied in kg despite being a 'weight' due to convention.

    Returns
    -------
    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.

    """

    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        self.add_input('CT', val=0.25, units='1/s')
        self.add_input('CL', val=0.7)
        self.add_input('CD', val=0.02)
        self.add_input('sonic_speed', val=100., units='m/s')
        self.add_input('rnge', val=3000., units='m')
        self.add_input('Mach_number', val=0.85)
        self.add_input('emptyTotal', val=120000., units='kg')

        self.add_output('fuelburn', val=1., units='kg')

        self.declare_partials('*', '*')
        self.set_check_partial_options(wrt='*', method='cs', step=1e-30)

    def compute(self, inputs, outputs):

        CT = inputs['CT']
        a = inputs['sonic_speed']
        rnge = inputs['rnge']
        M = inputs['Mach_number']
        emptyTotal = inputs['emptyTotal']

        CL = inputs['CL']
        CD = inputs['CD']

        outputs['fuelburn'] = (emptyTotal*4.45/9.81 + 42760) * (np.exp(rnge * CT / a / M * CD / CL) - 1)

    def compute_partials(self, inputs, partials):

        CT = inputs['CT']
        a = inputs['sonic_speed']
        rnge = inputs['rnge']
        M = inputs['Mach_number']
        emptyTotal = inputs['emptyTotal']


        CL = inputs['CL']
        CD = inputs['CD']

        dfb_dCL = -(emptyTotal) * np.exp(rnge * CT / a / M * CD / CL) \
            * rnge * CT / a / M * CD / CL ** 2
        dfb_dCD = (emptyTotal) * np.exp(rnge * CT / a / M * CD / CL) \
            * rnge * CT / a / M / CL
        dfb_dCT = (emptyTotal) * np.exp(rnge * CT / a / M * CD / CL) \
            * rnge / a / M / CL * CD
        dfb_drnge = (emptyTotal) * np.exp(rnge * CT / a / M * CD / CL) \
            / a / M / CL * CD * CT
        dfb_da = -(emptyTotal) * np.exp(rnge * CT / a / M * CD / CL) \
            * rnge * CT / a**2 / M * CD / CL
        dfb_dM = -(emptyTotal) * np.exp(rnge * CT / a / M * CD / CL) \
            * rnge * CT / a / M**2 * CD / CL

        dfb_dW = np.exp(rnge * CT / a / M * CD / CL) - 1

        partials['fuelburn', 'CL'] = dfb_dCL
        partials['fuelburn', 'CD'] = dfb_dCD
        partials['fuelburn', 'CT'] = dfb_dCT
        partials['fuelburn', 'sonic_speed'] = dfb_da
        partials['fuelburn', 'rnge'] = dfb_drnge
        partials['fuelburn', 'Mach_number'] = dfb_dM
        partials['fuelburn', 'emptyTotal'] = dfb_dW

