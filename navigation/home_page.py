import datetime
import os
import textwrap

# import stpyvista
import streamlit as st
import streamlit.components.v1 as components


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
        - Allow user to edit MOOSE input files on web (can upload already)
        - Launch from streamlit (this will be difficult to figure out)? 
            Or create a file that can be downloaded, then uploaded to HPC?

        </div>
    """
        ),
        unsafe_allow_html=True,
    )

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
    cmd_to_run = (
        "python utils/run_exodus_to_html_scene.py "
        "-i data/vac_meshed_oval_coil_and_stc.e "
        "-o pyvista.html "
        "-r metal "
    )
    if st.toggle("show_vacuum"):
        cmd_to_run += "-vac"
    os.system(cmd_to_run)

    HtmlFile = open("tmp_data/pyvista.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=300, width=600)