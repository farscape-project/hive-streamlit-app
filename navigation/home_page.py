import datetime
import os
import textwrap

import streamlit as st
import streamlit.components.v1 as components
# from stpyvista.trame_backend import stpyvista

from stpyvista.utils import start_xvfb

if "IS_XVFB_RUNNING" not in st.session_state:
  start_xvfb()
  st.session_state.IS_XVFB_RUNNING = True 
from stpyvista import stpyvista


def show_geom(inputfile, rendering, showvac=False):
    plotter = pv.Plotter(window_size=[400,400])

    # Create a mesh with a cube 
    mesh = pv.read(inputfile)
    if rendering.lower() == "metal":
        # plot coil with RGB for copper
        plotter.add_mesh(mesh.get(0)["coil"], color=[184, 115, 51])
        # plot target with RGB for copper
        plotter.add_mesh(mesh.get(0)["target"], color=[225,229,233])
        if show_vacuum:
            plotter.add_mesh(mesh.get(0)["vacuum_region"], opacity=0.2)
    elif rendering.lower() == "field":
        cmap = "plasma"
        plotter.add_mesh(mesh, cmap=cmap)
    elif rendering.lower() == "none":
        for block_name in mesh.get(0).keys():
            # check for vacuum, and avoid showing it
            if "vacuum" in block_name.lower():
                if show_vacuum:
                    plotter.add_mesh(mesh.get(0)[block_name], opacity=0.2)
            else:
                plotter.add_mesh(mesh.get(0)[block_name])

    # Final touches
    plotter.view_isometric()
    pv.global_theme.transparent_background = True


def home_page():
    st.divider()
    st.title("HIVE Web interface")
    st.markdown("")
    st.markdown("")
    st.markdown("**Overview**")
    st.markdown(
        textwrap.dedent(
            """\
        <div style="text-align: justify;">
        
        The Heating by Induction to Verify Extremes (HIVE) facility is 
        located at UKAEA's Culham site.
        It is designed to test thermo-mechanical and thermofluid performance
        of high heat flux components in a vacuum.
        Experimental results can also be used for verification and validation
        of simulations, which will provide complementary analysis.

        The geometry consists of a stainless steel monoblock with a coolant 
        pipe as shown below. The monoblock is heated by a 
        copper induction coil as shown below

        **TODO**:
        - Add more plots and data analysis to the `HIVE surrogate` page
        - Launch from streamlit (this will be difficult to figure out)? 
            Or create a file that can be downloaded, then uploaded to HPC?

        </div>
    """
        ),
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Welcome to the HIVE web-app")
        st.markdown(
            textwrap.dedent(
            """\
            <div style="text-align: justify;">

            This work was funded under the Fusion Computing Lab collaboration 
            between STFC Hartree Centre and UK Atomic Energy Authority.

            Developed by Josh Williams and Robert Gledhill 
            (STFC Hartree Centre).

            </div>
            """
            ),
            unsafe_allow_html=True,
        )
        st.image("images/UKAEA_WHITE_SML_AW.png")
        st.image("images/UKRI_STFC_HARTREE_CENTRE_WHITETEXT_RGB.png")

    # st.divider()

    # setup_mesh()
    # show_screenshot()

    """
    Pyvista and other vtk visualisers needs to be run on 
    main thread, which seems to be occupied by streamlit.
    We get an error like:
    *** Terminating app due to uncaught exception 
    'NSInternalInconsistencyException', 
    reason: 'NSWindow should only be instantiated on the main thread!'

    Workaround is to use os.system to run a script, then 
    read the data from that
    """

    inputfile = "data/vac_meshed_oval_coil_and_stc.e"
    rendering = "metal"
    show_vac = False
    cmd_to_run = (
        "python utils/run_exodus_to_html_scene.py "
        f"-i  {inputfile}"
        "-o pyvista.html "
        f"-r {rendering} "
    )
    if st.toggle("show_vacuum"):
        show_vac = True
        cmd_to_run += "-vac"
    # os.system(cmd_to_run)

    # HtmlFile = open("tmp_data/pyvista.html", "r", encoding="utf-8")
    # source_code = HtmlFile.read()
    # components.html(source_code, height=300, width=600)
    stpyvista(show_geom(inputfile, rendering, show_vac))
