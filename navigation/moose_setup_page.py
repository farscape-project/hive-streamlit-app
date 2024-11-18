import os
import textwrap
from glob import glob
from io import StringIO
from os import makedirs, path

import streamlit as st
import streamlit.components.v1 as components


def output_moose_txt(fname, contents):
    with st.expander(fname):
        # render moose file as code, using markdown syntax
        # need to add blank line at start and end, to avoid text being skipped
        st.markdown(f"```toml\n{contents}\n```")
    # st.divider()


def moose_setup_page():
    st.title("MOOSE file viewer")
    st.markdown(
        textwrap.dedent(
            """\
        <div style="text-align: justify;">
        
        Below we show the MOOSE baseline input files used to simulate HIVE.
        The fields available as sliders in the 'HIVE surrogate' tab were 
        randomly sampled from a uniform distribution to introduce 
        uncertainties to the simulation.

        We would like to introduce some way to modify the file on the web,
        and use this to create a UQ-toolkit configuration file. 
        These could then be launched on a HPC.
    
        </div>
    """
        ),
        unsafe_allow_html=True,
    )

    uploaded_file_list = st.file_uploader(
        "Optionally, replace the default MOOSE input files by your own:",
        accept_multiple_files=True,
    )
    if uploaded_file_list:
        # check if exodus mesh in file list
        pop_inds = []
        cmd_to_run = (
            "python utils/run_exodus_to_html_scene.py "
            "-i tmp_data/tmp.e "
            "-o tmp.html"
        )
        if st.toggle("show_vacuum"):
            cmd_to_run += " -vac"

        for i, uploaded_file in enumerate(uploaded_file_list):
            if ".e" in uploaded_file.name:
                makedirs("tmp_data/", exist_ok=True)
                with open("tmp_data/tmp.e", "wb") as f:
                    f.write(uploaded_file.getvalue())
                os.system(cmd_to_run)  # run python file
                HtmlFile = open("tmp_data/tmp.html", "r", encoding="utf-8")
                source_code = HtmlFile.read()
                components.html(source_code, height=300, width=600)
                pop_inds.append(i)
        # remove mesh files from list to avoid showing again,
        # do this by looping backwards over the list, to avoid accidentally
        # deleting data
        for pop_i in pop_inds[::-1]:
            uploaded_file_list.pop(pop_i)

        for uploaded_file in uploaded_file_list:
            if ".i" in uploaded_file.name:
                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = stringio.read()
                output_moose_txt(uploaded_file.name, string_data)
            # elif ".e" in uploaded_file.name:
            #     with open("tmp.e", "wb") as f:
            #         f.write(uploaded_file.getvalue())
            #     os.system("python utils/run_to_html.py tmp.e tmp.html")
            #     HtmlFile = open("tmp_data/tmp.html", 'r', encoding='utf-8')
            #     source_code = HtmlFile.read()
            #     components.html(source_code, height = 300, width=600)
    else:
        moose_files = glob(f"data/*.i")
        for file in moose_files:
            with open(file, "r") as f:
                lines = f.readlines()
            output_moose_txt(file.split("/")[-1], " ".join(lines))
