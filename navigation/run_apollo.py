"""Based on https://dunning-kruger.streamlit.app/"""
from __future__ import annotations

import os
import textwrap
from os.path import isfile
from typing import TYPE_CHECKING

import altair as alt
import hjson
import matplotlib.pyplot as plt
import numpy as np
try:
    import pyssam
except:
    print("pyssam not found")
import streamlit as st
import xgboost as xgb
import streamlit.components.v1 as components
import pyvista as pv
from stpyvista import stpyvista
from huggingface_hub import snapshot_download
from utils.run_exodus_to_html_scene import show_geom


from utils.inputFile_modifier import generate_modified_input_file
import subprocess
import sys

FNAME_PVTK = './OutputData/COMSOL_CrossVerification'
FNAME_electric_field_re  = './OutputData/electric_field_re.csv'
FNAME_joule_heating = './OutputData/joule_heating_density.csv'


Fname_electric_field_re = os.path.abspath( FNAME_electric_field_re )
Fname_joule_heating_density = os.path.abspath( FNAME_joule_heating )
Fname_paraview = os.path.abspath( FNAME_PVTK )

def write_field():

    mesh = pv.read(Fname_paraview) #readt the pvtk file
    plotter = pv.Plotter(window_size=[400,400])
    plotter.add_mesh(mesh, cmap="afmhot", scalars="joule_heating_density")

    del mesh
    
    plotter.view_isometric()
    pv.global_theme.transparent_background = True
    stpyvista( plotter )

def runSimulation(conductivity, current_magnitude, frequency):

    FNAME = './data/COMSOLValidationComplexAFormEM_1.i'; mpirun="38";

    new_fname = FNAME[:-2]+"_modified.i"

    Fname_electric_field_re = os.path.abspath( FNAME_electric_field_re )
    Fname_joule_heating_density = os.path.abspath( FNAME_joule_heating )
    Fname_paraview = os.path.abspath( FNAME_PVTK )

    parameters = dict(
        TargetEConductivity=conductivity, 
        CurrentMagnitude=current_magnitude, 
        Frequency=frequency,
        OutputFname_electric_field_re=Fname_electric_field_re,
        OutputFname_joule_heating_density= Fname_joule_heating_density,
        OutputFname_paraview = Fname_paraview
    )

    generate_modified_input_file( base_input_file=FNAME, new_input_filepath=new_fname, parameters=parameters )

    print("Successfully modify the file and saved at : ", new_fname)

    cmd = ["mpirun","-n", mpirun,  "/opt/apollo/apollo-opt", "-i", new_fname ]
    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True, bufsize=1)
    p.wait()

    print( "Finisshed running exit code :", p.returncode)    

    return p.returncode=="0"


def create_timeseries_plot(data):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Maximum temperature [K]")
    return fig


def custom_theme() -> dict[str, Any]:
    return {
        "config": {
            "axis": {
                "grid": False,
                "labelColor": "#7F7F7F",
                "labelFontSize": 14,
                "tickColor": "#7F7F7F",
                "titleColor": "#7F7F7F",
                "titleFontSize": 16,
                "titleFontWeight": "normal",
            },
            "legend": {
                "labelColor": "#7F7F7F",
                "labelFontSize": 14,
            },
            "view": {
                "height": 320,
                "width": 480,
                "stroke": False,
            },
        },
    }


def run_apollo():
    
    alt.themes.register("custom_theme", custom_theme)
    alt.themes.enable("custom_theme")

    with st.sidebar:
        st.header("KC4 Simulation Demonstration App")

        conductivity = float(st.text_input("Target Conductivity (S/m)", value=1.29e6))
        current_magnitude = float(st.text_input("Current Magnitude (A)", value=1000))
        frequency = float(st.text_input("Frequency (Hz):", value=1.0e5))

    
    result = runSimulation(conductivity, current_magnitude, frequency)
    if not result: #successful run
        return

    st.header("Temperature field at end of pulse")
    write_field()
