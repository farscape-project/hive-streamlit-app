import argparse
from os import makedirs
from sys import argv
import pyvista as pv


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

args = get_inputs()

plotter = pv.Plotter(window_size=[400,400])

# Create a mesh with a cube 
mesh = pv.read(args.inputfile)
if args.rendering.lower() == "metal":
    # plot coil with RGB for copper
    plotter.add_mesh(mesh.get(0)["coil"], color=[184, 115, 51])
    # plot target with RGB for copper
    plotter.add_mesh(mesh.get(0)["target"], color=[225,229,233])
    if args.show_vacuum:
        plotter.add_mesh(mesh.get(0)["vacuum_region"], opacity=0.2)
elif args.rendering.lower() == "field":
    cmap = "plasma"
    plotter.add_mesh(mesh, cmap=cmap)
elif args.rendering.lower() == "none":
    for block_name in mesh.get(0).keys():
        # check for vacuum, and avoid showing it
        if "vacuum" in block_name.lower():
            if args.show_vacuum:
                plotter.add_mesh(mesh.get(0)[block_name], opacity=0.2)
        else:
            plotter.add_mesh(mesh.get(0)[block_name])

# Final touches
plotter.view_isometric()
pv.global_theme.transparent_background = True
makedirs("tmp_data", exist_ok=True)
plotter.export_html(f'tmp_data/{args.outputfile}')
