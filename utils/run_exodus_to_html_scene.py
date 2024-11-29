import argparse
from os import makedirs
from sys import argv
import pyvista as pv
import streamlit as st


@st.cache_resource
def show_geom(inputfile, rendering, show_vacuum=False, render_counter=0):
    """
    st.cache_resource works when an input variable is changed only. 
    So to update the inference we add the render_counter. Otherwise, this is 
    always 0.

    Parameters
    ----------
    - 
    -
    -
    render_counter : int
        Dummy parameter to fix caching
    """
    plotter = pv.Plotter(window_size=[400,400])
    print(f"render counter = {render_counter}")

    # Create a mesh with a cube 
    if type(inputfile) == str:
        mesh = pv.read(inputfile)
    else:
        print("input file is pyvista object")
        mesh = inputfile
    if rendering.lower() == "metal":
        # plot coil with RGB for copper
        plotter.add_mesh(mesh.get(0)["coil"], color=[184, 115, 51])
        # plot target with RGB for copper
        plotter.add_mesh(mesh.get(0)["target"], color=[225,229,233])
        if show_vacuum:
            plotter.add_mesh(mesh.get(0)["vacuum_region"], opacity=0.2)
    elif rendering.lower() == "field":
        cmap = "coolwarm"
        plotter.add_mesh(mesh, cmap=cmap, scalars="Temperature [K]")
    elif rendering.lower() == "none":
        for block_name in mesh.get(0).keys():
            # check for vacuum, and avoid showing it
            if "vacuum" in block_name.lower():
                if show_vacuum:
                    plotter.add_mesh(mesh.get(0)[block_name], opacity=0.2)
            else:
                plotter.add_mesh(mesh.get(0)[block_name])

    del mesh
    # Final touches
    plotter.view_isometric()
    pv.global_theme.transparent_background = True
    return plotter


def get_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        "-i",
        required=True,
        type=str,
        help="which mesh file to read?",
    )
    parser.add_argument(
        "--outputfile",
        "-o",
        required=True,
        type=str,
        help="which html file to write in `tmp_data/` dir?",
    )
    parser.add_argument(
        "--rendering",
        "-r",
        type=str,
        choices=["metal", "none", "field"],
        default="none",
        help="Show colours for 'target' and 'coil' blocks",
    )
    parser.add_argument(
        "--show-vacuum",
        "-vac",
        action="store_true",
        default=False,
        help="Show vacuum",
    )
    return parser.parse_args()


if __name__ == "__main__":

    plotter = pv.Plotter(window_size=[400,400])

    args = get_inputs()
    plotter = show_geom(args.inputfile, args.rendering, args.show_vacuum)
    pv.global_theme.transparent_background = True
    makedirs("tmp_data", exist_ok=True)
    plotter.export_html(f'tmp_data/{args.outputfile}')
    del plotter
