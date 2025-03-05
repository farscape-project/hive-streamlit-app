#-------------------------------------------------------------------------
# PARAMETER DEFINITIONS 
#_*
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Geometry
OutputFname_electric_field_re = '/opt/apollo/apollo_gui_python/back-end/OutputData/electric_field_re.csv'
OutputFname_joule_heating_density = '/opt/apollo/apollo_gui_python/back-end/OutputData/joule_heating_density.csv'
OutputFname_paraview = '/opt/apollo/apollo_gui_python/back-end/OutputData/COMSOL_CrossVerification'
Frequency=1.0e5
TargetEConductivity=1.29e6
CurrentMagnitude=1000.0
#**
#-------------------------------------------------------------------------
[Mesh]
  type = CoupledMFEMMesh
  file = ./Meshing/vac_meshed_comsol_coil_and_target.e
  dim = 3
[]

[Problem]
  type = MFEMProblem
[]

[Formulation]
  type = ComplexAFormulation
  magnetic_vector_potential_name = magnetic_vector_potential
  magnetic_vector_potential_re_name = magnetic_vector_potential_re
  magnetic_vector_potential_im_name = magnetic_vector_potential_im
  frequency_name = frequency
  magnetic_reluctivity_name = magnetic_reluctivity
  magnetic_permeability_name = magnetic_permeability
  electric_conductivity_name = electrical_conductivity
  dielectric_permittivity_name = dielectric_permittivity

  electric_field_re_name = electric_field_re
  electric_field_im_name = electric_field_im
  current_density_re_name = current_density_re
  current_density_im_name = current_density_im
  magnetic_flux_density_re_name = magnetic_flux_density_re
  magnetic_flux_density_im_name = magnetic_flux_density_im
  joule_heating_density_name = joule_heating_density
[]

[FESpaces]
  [H1FESpace]
    type = MFEMFESpace
    fespace_type = H1
    order = FIRST
  []
  [HCurlFESpace]
    type = MFEMFESpace
    fespace_type = ND
    order = FIRST
  []
  [HDivFESpace]
    type = MFEMFESpace
    fespace_type = RT
    order = CONSTANT
  []
[]

[AuxVariables]
  [magnetic_vector_potential_re]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [magnetic_vector_potential_im]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [electric_field_re]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [electric_field_im]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [current_density_re]
    type = MFEMVariable
    fespace = HDivFESpace
  []
  [current_density_im]
    type = MFEMVariable
    fespace = HDivFESpace
  []
  [magnetic_flux_density_re]
    type = MFEMVariable
    fespace = HDivFESpace
  []
  [magnetic_flux_density_im]
    type = MFEMVariable
    fespace = HDivFESpace
  []

  [source_current_density]
    type = MFEMVariable
    fespace = HDivFESpace
  []
  [source_electric_field]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [source_electric_potential]
    type = MFEMVariable
    fespace = H1FESpace
  []
  [joule_heating_density]
    family = MONOMIAL
    order = FIRST
    initial_condition = 0.0
  []
[]

[BCs]
  [tangential_E_bdr]
    type = MFEMComplexVectorDirichletBC
    variable = magnetic_vector_potential
    boundary = '1 2 3'
    real_vector_coefficient = TangentialECoef
    imag_vector_coefficient = TangentialECoef
  []
[]

[Materials]  
  [target]
    type = MFEMConductor
    electrical_conductivity_coeff = TargetEConductivity
    electric_permittivity_coeff = VoidPermittivity
    magnetic_permeability_coeff = VoidPermeability
    block = 2
  []
  [void]
    type = MFEMConductor
    electrical_conductivity_coeff = VoidEConductivity
    electric_permittivity_coeff = VoidPermittivity
    magnetic_permeability_coeff = VoidPermeability
    block = '1 3 4'
  []
[]

[VectorCoefficients]
  [TangentialECoef]
    type = MFEMVectorConstantCoefficient
    value_x = 0.0
    value_y = 0.0
    value_z = 0.0
  []
[]

[Coefficients]
  [CoilEConductivity]
    type = MFEMConstantCoefficient
    value = 5.998e7 # S/m 
  []  
  [VoidEConductivity]
    type = MFEMConstantCoefficient
    value = 1.0 # S/m
  []
  [VoidPermeability]
    type = MFEMConstantCoefficient
    value = 1.25663706e-6 # T m/A
  []
  [VoidPermittivity]
    type = MFEMConstantCoefficient
    value = 8.85418782e-12
  []

  [TargetEConductivity]
    type = MFEMConstantCoefficient
    value = ${TargetEConductivity} #1.29e6 # S/m 
  []

  # 1 kA RMS current at 100 kHz
  [CurrentMagnitude]
    type = MFEMConstantCoefficient
    value = ${CurrentMagnitude} #1000.0
  []
  [frequency]
    type = MFEMConstantCoefficient
    value = ${Frequency} 
  []
[]

[Sources]
  [SourcePotential]
    type = MFEMOpenCoilSource
    total_current_coef = CurrentMagnitude
    # electrical_conductivity_coef = CoilEConductivity
    source_current_density_gridfunction = source_current_density
    source_electric_field_gridfunction = source_electric_field
    source_potential_gridfunction = source_electric_potential
    coil_in_boundary = 1
    coil_out_boundary = 2
    block = '3 4'

    l_tol = 1e-10
    l_abs_tol = 1e-10
    l_max_its = 1000
    print_level = 1
  []
[]

[Executioner]
  type = Steady
  l_tol = 1e-8
  l_abs_tol = 1e-8
  l_max_its = 1000

[]

# [AuxKernels]
#   [LineSampler]
#     type = MFEMLineSamplerAux
#     filename = ${OutputFname_electric_field_re}
#     variable = electric_field_re
#     num_points = 100
#     start_point = '-0.0114445 -5e-6 0.019444'
#     end_point = '0.0114445 -5e-6 0.019444'
#     header = 't (s), x (m), y (m), z (m), E_x (V/m), E_y (V/m), E_z (V/m)'
#   []
#   [LineSampler2]
#     type = MFEMLineSamplerAux
#     filename = ${OutputFname_joule_heating_density}
#     variable = joule_heating_density
#     num_points = 100
#     start_point = '-0.0114445 -5e-6 0.019444'
#     end_point = '0.0114445 -5e-6 0.019444'
#     header = 't (s), x (m), y (m), z (m), Q(watts/cbm)'
#   []
# []

[Outputs]
  [ParaViewDataCollection]
    type = MFEMParaViewDataCollection
    file_base = ${OutputFname_paraview}
    high_order_output = true
  []
  exodus=true
[]

