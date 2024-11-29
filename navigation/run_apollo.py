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

FNAME_PVTK = './OutputData/COMSOL_CrossVerification/'
FNAME_electric_field_re  = './OutputData/electric_field_re.csv'
FNAME_joule_heating = './OutputData/joule_heating_density.csv'


Fname_electric_field_re = os.path.abspath( FNAME_electric_field_re )
Fname_joule_heating_density = os.path.abspath( FNAME_joule_heating )
Fname_paraview = os.path.abspath( FNAME_PVTK )

def runSimulation(conductivity, current_magnitude, frequency):

    mpirun="1"
    # FNAME = './data/COMSOLValidationComplexAFormEM_1.i'
    FNAME = './data/COMSOLValidationComplexAFormEM_1.i'; 

    new_fname = FNAME[:-2]+"_modified.i"

    run_fname = "./data/COMSOLValidationDummy.i"
    # run_fname = new_fname

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


    cmd = ["mpirun","-n", mpirun,  "/opt/apollo/apollo-opt", "-i", run_fname, "-w" ]
    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True, bufsize=1)
    p.wait()

    print("Finished running. exit code :", p.returncode)    

    return p.returncode == 0


def create_timeseries_plot(data):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Maximum temperature [K]")
    return fig


def old_write_field(field_name, fname_in):
    """
    Use pyvista to write field values (from our surrogate) to a template
    exodus mesh. The scene is then rendered and saved as html, which is read
    by streamlit

    Parameters
    ----------
    field_name : str
        Name of field to show on scale bar in streamlit app
    fname_in : str
        Name of exodus file for reading
    """

    cmd_to_run = (
        "python utils/run_exodus_to_html_scene.py "
        f"-i {fname_in} "
        "-o apollo.html "
        "-r field-apollo "
    )
    os.system(cmd_to_run)

    HtmlFile = open("tmp_data/apollo.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=600, width=600)

def run_apollo():
    st.title("Simulation of HIVE with Apollo")

    st.markdown(
        textwrap.dedent(
            """\
        <div style="text-align: justify;">

        This mini-app uses a coarse mesh of the HIVE geometry to simulate
        the heating deposited onto the target. This is done using Apollo.

        The advantages of this compared to the surrogate on the previous slide
        is its ability to extrapolate to a broader range of (possibly unseen) 
        parameters.

        </div>
    """
        ),
        unsafe_allow_html=True,
    )



    with st.sidebar:
        st.header("FCL and KC4 Simulation Demonstration App")

        conductivity = float(st.text_input("Target Conductivity (S/m)", value=1.29e6))
        current_magnitude = float(st.text_input("Current Magnitude (A)", value=1000))
        frequency = float(st.text_input("Frequency (Hz):", value=1.0e5))
        button = st.button("Submit", type="primary")

    simworked = True
    if button:
        simworked = runSimulation(conductivity, current_magnitude, frequency)
        button = False

    print(simworked)
    st.divider()
    Fname_paraview = "./data/COMSOLValidationDummy_out.e"
    if simworked:
        if os.path.isfile(Fname_paraview):
            st.header("Joule heating density (coarse mesh)")
            old_write_field("joule_heating_density", Fname_paraview)
            print("Found results", Fname_paraview)
            os.remove("tmp_data/apollo.html")
        else:
            st.markdown("No results found")
    else:
        st.markdown("**Simulation failed. This usually happens when frequency < 1e5**")
    