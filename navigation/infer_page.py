"""Based on https://dunning-kruger.streamlit.app/"""
from __future__ import annotations

import os
import textwrap
from os.path import isfile
from typing import TYPE_CHECKING

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pyssam
import streamlit as st
import xgboost as xgb
import streamlit.components.v1 as components
import pyvista as pv
from huggingface_hub import snapshot_download


def write_field(field_name, field_val, fname_in, fname_out):
    """
    Use pyvista to write field values (from our surrogate) to a template
    exodus mesh. The scene is then rendered and saved as html, which is read
    by streamlit

    Parameters
    ----------
    field_name : str
        Name of field to show on scale bar in streamlit app
    field_val : array_like
        Scalar field values for writing to points in exodus app
    fname_in : str
        Name of template exodus file for writing new data to.
        Currently, this is stored on our huggingface repo.
    fname_out : str
        Name of vtk file to write new field data to (this is read by renderer) 
    """
    m = pv.read(fname_in)
    c = m.get(0)[0]
    c.point_data.set_scalars(field_val, field_name)
    c.save(fname_out)

    cmd_to_run = (
        "python utils/run_exodus_to_html_scene.py "
        f"-i {fname_out} "
        "-o field.html "
        "-r field "
    )
    os.system(cmd_to_run)

    HtmlFile = open("tmp_data/field.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=600, width=600)


class Reconstructor:
    def __init__(self, xgb_fname, pod_coefs_fname):
        """
        Temporary copy of a class from uq_toolkit
        (until uq_toolkit public for installation)
        """
        self._load_xgb_regressor(xgb_fname)
        self._load_pyssam(pod_coefs_fname)

    def _load_xgb_regressor(self, xgb_fname):
        """
        Read file with XGBoost config and weights.
        Prerequisite is that `train_xgb.py` has been run already.

        Parameters
        ----------
        xgb_fname : str
            /path/to/xgb_model.bin
        """
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(xgb_fname)

    def _load_pyssam(self, pyssam_fname):
        """
        Create a dummy pyssam object, and read file with POD information.
        Prerequisite is that `find_pod_modes.py` has been run already.

        Parameters
        ----------
        pyssam_fname : str
            /path/to/pod_data.npz
        """
        # TODO: Implement such that object can be used with no dataset
        # i.e. train offline and use obj at inference time
        # make pyssam.morph_model a staticmethod
        self.sam_obj = pyssam.SAM(
            np.random.normal(size=(3, 3))
        )  # create dummy sam_obj
        npzfile = np.load(pyssam_fname)
        self.mean_dataset_columnvector = npzfile["mean"]
        self.pca_model_components = npzfile["pca_components"]
        self.sam_obj.std = npzfile["pca_std"]

    def reconstruct_with_xgboost(
        self, t, param_list, reduction=None, num_modes=2
    ):
        """
        Reconstruct a field using POD pre-defined modes, and mode coefficients
        determined by an xgboost-regression model.
        The mode coefficients are based on time, t, and some parameters.

        Parameters
        ----------
        t : float
            physical time value to reconstruct
        param_list : list
            parameters needed for doing inference on the xgboost model
            ordering must be same as defined during training.
        reduction : function or None
            optional operation to apply to data such as np.max, np.mean, when only
            a single value is needed
        num_modes : int
            number of POD modes to use in reconstruction

        Returns
        -------
        recon_field : array_lie
            Reconstructed field values (or, optionally reduced to scalar)
        """
        feat_mat = xgb.DMatrix([[t, *param_list]])
        pod_coefs = np.array(self.xgb_model.predict(feat_mat)).squeeze()

        # fix for when num_modes > pod_coefs
        num_modes = min(len(pod_coefs), num_modes)

        recon_field = self.sam_obj.morph_model(
            self.mean_dataset_columnvector,
            self.pca_model_components,
            pod_coefs[:num_modes],
            num_modes=num_modes,
        )
        if reduction is not None:
            return reduction(recon_field)
        else:
            return recon_field


def generate_data(coeff_list, xgb_file, pod_file, num_modes):
    recon_model = Reconstructor(xgb_file, pod_file)
    time_list = np.arange(5, 61, 5)  # TODO: fix hard-codeness
    out_temp = []
    for t in time_list:
        # do inference
        data_out = recon_model.reconstruct_with_xgboost(
            t, coeff_list, reduction=None, num_modes=num_modes
        )

        out_temp.append(data_out.max())
    return np.c_[time_list, out_temp], data_out


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


def application_page():
    alt.themes.register("custom_theme", custom_theme)
    alt.themes.enable("custom_theme")

    with st.sidebar:
        st.header("Surrogate Parameters")

        num_modes = st.slider(
            label="Number of POD modes",
            min_value=2,
            max_value=5,
            value=4,
            step=1,
        )

        st.divider()
        st.header("Physical Parameters")

        htc_coef = st.slider(
            label="Pipe heat transfer coefficient (scaling factor)",
            min_value=0.9,
            max_value=1.1,
            value=1.0,
            step=0.01,
        )
        e_cond = st.slider(
            label="Electrical conductivity (scaling factor)",
            min_value=0.9,
            max_value=1.1,
            value=1.0,
            step=0.01,
        )
        current = st.slider(
            label="Current",
            min_value=900.0,
            max_value=1100.0,
            value=1000.0,
            step=1.0,
        )

    MODEL_DIR = "tmp_data"
    REGRESSION_MODEL_NAME = "xgb_model.bin"
    REGRESSION_MODEL_PATH = f"{MODEL_DIR}/{REGRESSION_MODEL_NAME}"
    SPATIAL_MODEL_NAME = "pod_weights_truncated.npz"
    SPATIAL_MODEL_PATH = f"{MODEL_DIR}/{SPATIAL_MODEL_NAME}"

    if isfile(REGRESSION_MODEL_PATH) and isfile(SPATIAL_MODEL_PATH):
        pass
    else:
        snapshot_download(
            repo_id="jvwilliams23/hive-xgb",
            # allow_patterns=[REGRESSION_MODEL_NAME, SPATIAL_MODEL_NAME],
            local_dir=f"{MODEL_DIR}/",
        )

    tab_data, field_vals = generate_data(
        coeff_list=[htc_coef, current, e_cond],
        xgb_file=REGRESSION_MODEL_PATH,
        pod_file=SPATIAL_MODEL_PATH,
        num_modes=num_modes,
    )

    st.title("Online inference of HIVE experiments")

    st.markdown(
        textwrap.dedent(
            """\
        <div style="text-align: justify;">

        This mini-app uses a surrogate model trained on temperature field 
        snapshots of HIVE simulations performed using MOOSE with Apollo.

        The surrogate model reconstructs the full temperature field using
        principal component analysis (PCA) to learn the spatial structures 
        in the snapshot data. 
        The time and parameter dependence of the PCA components is learned 
        using XGBoost, which is a decision tree algorithm, known to work well
        on tabular data.

        We chose XGBoost due to the simple nature of the data, but this can 
        easily be replaced with e.g. a neural network defined in pytorch.

        **TODO:** add more analysis e.g. thermocouples, sensitivity, visualisations
        </div>
    """
        ),
        unsafe_allow_html=True,
    )

    st.header("Timeseries plot of HIVE pulse")
    st.pyplot(
        create_timeseries_plot(tab_data),
    )

    write_field(
        "Temperature [K]",
        field_vals,
        "tmp_data/example_moose_output_temperature_out.e",
        "tmp_data/temp_field.vtk",
    )
