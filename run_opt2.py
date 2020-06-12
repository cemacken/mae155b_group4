from __future__ import division, print_function
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizeDriver, SqliteRecorder
from lsdo_utils.api import LinearCombinationComp, PowerCombinationComp
from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant
from components.zero_lift_drag.atmosphere_group import AtmosphereGroup


shape = (1,)
# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 5,
             'num_x' : 2,
             'wing_type' : 'CRM',
             'symmetry' : True,
             'num_twist_cp' : 5}

mesh, twist_cp = generate_mesh(mesh_dict)

surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'thickness_cp' : np.array([.1, .2, .3]),

            'twist_cp' : twist_cp,
            'mesh' : mesh,

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : 0.2,            # CL of the surface at alpha=0
            'CD0' : 0.013,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c_cp' : np.array([0.14]),      # thickness over chord ratio (NACA0015)
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                    # thickness

            'with_viscous' : True,
            'with_wave' : True,     # if true, compute wave drag

            # Structural values are based on aluminum 7075
            'E' : 70.e9,            # [Pa] Young's modulus of the spar
            'G' : 30.e9,            # [Pa] shear modulus of the spar
            'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
            'mrho' : 3.e3,          # [kg/m^3] material density
            'fem_origin' : 0.35,    # normalized chordwise location of the spar
            'wing_weight_ratio' : 2.,
            'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
            'distributed_fuel_weight' : False,
            # Constraints
            'exact_failure_constraint' : False, # if false, use KS function
            }

# Create the problem and assign the model group
prob = Problem()

# Add problem information as an independent variables component
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=250, units='m/s')
indep_var_comp.add_output('alpha', val=2.5, units='deg')
indep_var_comp.add_output('Mach_number', val=0.84)
indep_var_comp.add_output('re', val=1.e6, units='1/m')
indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
indep_var_comp.add_output('CT', val= grav_constant * 17.e-6, units='1/s') # Specific fuel consumption for the entire aircraft. If using ISP input 1/ISP value
indep_var_comp.add_output('R', val=1.3e6, units='m') # range
# The operating empty weight of the aircraft, without fuel or structural mass. Supplied in kg despite being a 'weight' due to convention.
indep_var_comp.add_output('W0', val=54412.4,  units='kg') 
indep_var_comp.add_output('speed_of_sound', val=295.4, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)
indep_var_comp.add_output('empty_cg', val=np.zeros((3)), units='m')
# some wing variables to add
indep_var_comp.add_output('sweep', val=25, units='deg')
indep_var_comp.add_output('span', val=57, units='m')
indep_var_comp.add_output('taper', val=0.3)
indep_var_comp.add_output('dihedral', val=3, units='deg')
indep_var_comp.add_output('S_ref_total', val=430, units='m**2')
# indep_var_comp.add_output('S_ref', val=360, units='m**2')


prob.model.add_subsystem('flight_and_wing_properties',
     indep_var_comp,
     promotes=['*'])

# atmosphere_group = AtmosphereGroup(shape = shape,)
# prob.model.add_subsystem('atmosphere_group', atmosphere_group, promotes=['*'])

aerostruct_group = AerostructGeometry(surface=surface)

name = 'wing'

# Add tmp_group to the problem with the name of the surface.
prob.model.add_subsystem(name, aerostruct_group)

point_name = 'AS_point_0'

# Create the aero point group and add it to the model
AS_point = AerostructPoint(surfaces=[surface], user_specified_Sref=True)

prob.model.add_subsystem(point_name, AS_point,
    promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'CT', 'R',
        'W0', 'speed_of_sound', 'empty_cg', 'load_factor'])

com_name = point_name + '.' + name + '_perf'
prob.model.connect(name + '.local_stiff_transformed', point_name + '.coupled.' + name + '.local_stiff_transformed')
prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')

# Connect aerodyamic mesh to coupled group mesh
prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

# Connect performance calculation variables
prob.model.connect(name + '.radius', com_name + '.radius')
prob.model.connect(name + '.thickness', com_name + '.thickness')
prob.model.connect(name + '.nodes', com_name + '.nodes')
prob.model.connect(name + '.cg_location', point_name + '.' + 'total_perf.' + name + '_cg_location')
prob.model.connect(name + '.structural_mass', point_name + '.' + 'total_perf.' + name + '_structural_mass') # in N
prob.model.connect(name + '.t_over_c', com_name + '.t_over_c')

# prob.model.connect('S_ref','AS_point_0.wing_perf.S_ref')
prob.model.connect('span','wing.geometry.mesh.stretch.span')
prob.model.connect('sweep','wing.geometry.mesh.sweep.sweep')
prob.model.connect('taper','wing.geometry.mesh.taper.taper')
prob.model.connect('dihedral','wing.geometry.mesh.dihedral.dihedral')
prob.model.connect('S_ref_total', point_name + '.S_ref_total')

comp = ExecComp('aspect_ratio = span**2 / S_ref_total')
prob.model.add_subsystem('aspect_ratio_comp', comp, promotes=['*'])

prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['tol'] = 1e-9
prob.driver.options['debug_print'] = ['nl_cons','objs', 'desvars']
# adds recorder
# recorder = SqliteRecorder("aerostruct.db")
# prob.driver.add_recorder(recorder)
# prob.driver.recording_options['record_derivatives'] = True
# prob.driver.recording_options['includes'] = ['*']

# Setup problem and add design variables, constraint, and objective
# prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
# prob.model.add_design_var('wing.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
# prob.model.add_constraint('AS_point_0.wing_perf.failure', upper=0.)
prob.model.add_constraint('AS_point_0.wing_perf.thickness_intersects', upper=0.)

# Add design variables, constraisnt, and objective on the problem
prob.model.add_design_var('span', lower=50,upper=62)
prob.model.add_design_var('sweep',lower=20, upper=30)
prob.model.add_design_var('S_ref_total',lower=300, upper=450)
# prob.model.add_design_var('dihedral',lower=0, upper=3)
# prob.model.add_design_var('alpha', lower=-10, upper=10)

# prob.model.add_constraint('alpha',lower=0, upper=3)
prob.model.add_constraint('AS_point_0.L_equals_W', lower=-1e-3, upper=1e-3)

prob.model.add_objective('AS_point_0.fuelburn', scaler = -1e-5)

# Set up the problem
prob.setup(check=True)

from openmdao.api import n2
n2(prob)
# Only run analysis

prob.run_driver()
# prob.run_model()
prob.model.list_inputs(prom_name=True)
# prob.model.list_outputs(prom_name=True)

# Run optimization

print()
print('aspect_ratio:', prob['aspect_ratio'])
print('span:', prob['span'])
print('S_w:', prob['S_ref_total'])
print('alpha:', prob['alpha'])
print('CL:', prob['AS_point_0.wing_perf.CL'])
print('CD:', prob['AS_point_0.wing_perf.CD'])
print('LD:', prob['AS_point_0.wing_perf.CL'] / prob['AS_point_0.wing_perf.CD'])
# print('altitude:', prob['altitude'])