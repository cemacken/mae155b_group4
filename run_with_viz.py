import numpy as np

from openmdao.api import Group, IndepVarComp, ExecComp, ScipyOptimizeDriver
from lsdo_utils.api import PowerCombinationComp, LinearPowerCombinationComp
from openaerostruct.geometry.utils import generate_mesh
from components.oas_group import OASGroup
from components.breguet_range.breguet_range_comp import BregRangeCo
from components.zero_lift_drag.zero_lift_group import ZeroLiftGroup
from components.aeroprop.thrust_comp import thrustComp
from components.aeroprop.drag_comp import dragComp
from components.aeroprop.lift_comp import liftComp
from weight_component.weightGroup import weightCompGroup
from components.zero_lift_drag.atmosphere_group import AtmosphereGroup
from lsdo_viz.api import Problem

shape = (1,)
# Create a dictionary to store options about the mesh
mesh_dict = {'num_y' : 11,
             'num_x' : 5,
             'wing_type' : 'CRM',
             'symmetry' : False,
             'num_twist_cp' : 3}

# Generate the aerodynamic mesh based on the previous dictionary
mesh, twist_cp = generate_mesh(mesh_dict)

# Create a dictionary with info and options about the aerodynamic
# lifting surface

# Create the OpenMDAO problem
prob = Problem()

model = Group()

comp = IndepVarComp()
# comp.add_output('speed', val=257.22)
comp.add_output('rnge', val=1.3e6)
comp.add_output('isp', val=10193) #dummy variable for now
comp.add_output('altitude', val = 10000) # in meters
comp.add_output('characteristic_length', val = 5)
# comp.add_output('S_w', val = 157)
# wing parameters
comp.add_output('span', val = 59)
comp.add_output('aspect_ratio', val = 9)
comp.add_output('dihedral', val = 3, units='deg')

# comp.add_output('BPR', val = 5) # Bypass ratio
# comp.add_output('max_thrust', val = 490)

# Add vars to model, promoting is a quick way of automatically connecting inputs
# and outputs of different OpenMDAO components
prob.model.add_subsystem('flight_vars', comp, promotes=['*'])

comp = ExecComp('S_w = span**2 / aspect_ratio')
prob.model.add_subsystem('wing_area_comp', comp, promotes=['*'])

comp = ExecComp('chord = S_w / span')
prob.model.add_subsystem('chord_length_comp', comp, promotes=['*'])

atmosphere_group = AtmosphereGroup(shape = shape,)
prob.model.add_subsystem('atmosphere_group', atmosphere_group, promotes=['*'])

surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : False,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'twist_cp' : twist_cp,
            'mesh' : mesh,

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : 0.2,            # CL of the surface at alpha=0
            'CD0' : .013,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c_cp' : np.array([0.14]),      # thickness over chord ratio (NACA0014)
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                    # thickness
            'with_viscous' : True,  # if true, compute viscous drag
            'with_wave' : True,     # if true, compute wave drag
            }

oas_group = OASGroup(surface=surface)
prob.model.add_subsystem('oas_group', oas_group, promotes=['*'])

comp = weightCompGroup()
prob.model.add_subsystem('weight_group', comp, promotes=['*'])

comp = BregRangeCo() 
prob.model.add_subsystem('breguet_range_comp', comp, promotes=['*'])

# comp = dragComp() # this component computes drag from CD found in OpenAeroStruct
# prob.model.add_subsystem('drag_comp', comp, promotes=['*'])

# comp = liftComp() # this component computes lift from CL found in OpenAeroStruct
# prob.model.add_subsystem('lift_comp', comp, promotes=['*'])

# comp = thrustComp() 
# prob.model.add_subsystem('thrust_comp', comp, promotes=['*'])

# Computes E = L/D for use in the breguet range component
comp = ExecComp('LD = CL/CD')
prob.model.add_subsystem('ld_comp', comp, promotes=['*'])

# comp = ExecComp('TOD = thrust - drag ')
# prob.model.add_subsystem('TOD', comp, promotes=['*'])

comp = LinearPowerCombinationComp(
    shape=shape,
    out_name = 'LOW',
    terms_list=[
        (1.0, dict(
            L = 1.,  
        )),
        (-9.81, dict(
            emptyTotal = 1,
        )),
    ],
)
prob.model.add_subsystem('LOW', comp, promotes=['*'])


prob.model.connect('aero_point_0.CL', 'CL')
prob.model.connect('aero_point_0.CD', 'CD')
prob.model.connect('aero_point_0.wing_perf.L', 'L')
prob.model.connect('dihedral', 'wing.mesh.dihedral.dihedral')
# prob.model.connect('chord', 'wing.mesh.scale_x.chord'[0])


# Import the Scipy Optimizer and set the driver of the problem to use
# it, which defaults to an SLSQP optimization method

# # recorder = om.SqliteRecorder("aero.db")
# # prob.driver.add_recorder(recorder)
# # prob.driver.recording_options['record_derivatives'] = True
# # prob.driver.recording_options['includes'] = ['*']

# # Set optimizer as model driver
prob.driver = ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'COBYLA'
# prob.driver.options['tol'] = 1e-9
# prob.driver.options['debug_print'] = ['nl_cons','objs', 'desvars']

# # Setup problem and add design variables, constraint, and objective
prob.model.add_design_var('alpha', lower=-5, upper = 15)
prob.model.add_design_var('altitude_km', lower=10, upper = 15)
# # Constraints

# # prob.model.add_constraint('aero_point_0.CL', equals = 0.7)
# # prob.model.add_constraint('TOD', equals = 0)
prob.model.add_constraint('LOW', lower=-1e-3, upper=1e-3, scaler=1e-6)

# # Objective

prob.model.add_objective('W_f', scaler=-1)


# Set up and run the optimization problem
prob.setup()

# prob.check_partials(compact_print=True)
# exit()


# # # Run optimization

prob.run()

# prob.run_model()

# prob.model.list_inputs(prom_name=True)
# prob.model.list_outputs(prom_name=True)
