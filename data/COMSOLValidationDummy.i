CoolantBulkTemperature = 300. # K

[Mesh]
  [meshed-coil-and-target]
    type = FileMeshGenerator
    file = ./Meshing/vac_meshed_comsol_coil_and_target.e
  []
#   second_order = true
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = FIRST
    initial_condition = ${CoolantBulkTemperature}
  []
[]

[Kernels]
  [diff]
    type = Diffusion
    variable = temperature
  []
[]

[AuxVariables]
  [joule_heating_density]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[Executioner]
  automatic_scaling = true
  solve_type = 'NEWTON'
  type = Transient
  dt = 5.0
  start_time = 0.0
  end_time = 0.0
[]

[Outputs]
  exodus = true
[]

[MultiApps]
  [sub_app]
    type = FullSolveMultiApp
    positions = '0 0 0'
    input_files = 'COMSOLValidationComplexAFormEM_1_modified.i'
    execute_on = INITIAL
  []
[]

[Transfers]
  [pull_joule_heating]
    type = MultiAppGeneralFieldShapeEvaluationTransfer

    # Transfer from the sub-app to this app
    from_multi_app = sub_app

    # The name of the variable in the sub-app
    source_variable = joule_heating_density

    # The name of the auxiliary variable in this app
    variable = joule_heating_density
  []
[]