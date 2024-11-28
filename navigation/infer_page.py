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
import platform
from utils.run_exodus_to_html_scene import show_geom


def old_write_field(field_name, field_val, fname_in, fname_out):
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

def write_field(field_name, field_val, fname_in, fname_out):
    m = pv.read(fname_in)
    c = m.get(0)[0]
    c.point_data.set_scalars(field_val, field_name)
    c.save(fname_out)
    del m, c, field_val
    stpyvista(show_geom(fname_out, rendering="field"))


class Reconstructor:
    def __init__(_self, xgb_fname, pod_coefs_fname):
        """
        Temporary copy of a class from uq_toolkit
        (until uq_toolkit public for installation)
        """
        _self.sam_obj = pyssam.SAM(
            np.random.normal(size=(3, 3))
        )  # create dummy sam_obj
        _self.xgb_model = _self._load_xgb_regressor(xgb_fname)
        _self.mean_dataset_columnvector, _self.pca_model_components, _self.sam_obj.std = _self._load_pyssam(pod_coefs_fname)

    @st.cache_resource
    def _load_xgb_regressor(_self, xgb_fname):
        """
        Read file with XGBoost config and weights.
        Prerequisite is that `train_xgb.py` has been run already.

        Parameters
        ----------
        xgb_fname : str
            /path/to/xgb_model.bin
        """
        xgb_model = xgb.Booster()
        xgb_model.load_model(xgb_fname)
        return xgb_model

    @st.cache_data
    def _load_pyssam(_self, pyssam_fname):
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

        npzfile = np.load(pyssam_fname)
        return npzfile["mean"], npzfile["pca_components"], npzfile["pca_std"]

    def reconstruct_with_xgboost(
        _self, t, param_list, reduction=None, num_modes=2
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
        pod_coefs = np.array(_self.xgb_model.predict(feat_mat)).squeeze()

        # fix for when num_modes > pod_coefs
        num_modes = min(len(pod_coefs), num_modes)

        recon_field = _self.sam_obj.morph_model(
            _self.mean_dataset_columnvector,
            _self.pca_model_components,
            pod_coefs[:num_modes],
            num_modes=num_modes,
        )
        if reduction is not None:
            return reduction(recon_field)
        else:
            return recon_field


def generate_data(coeff_list, xgb_file, pod_file, num_modes, slider_time=60.0):
    recon_model = Reconstructor(xgb_file, pod_file)
    time_list = np.arange(5, 61, 5)  # TODO: fix hard-codeness
    out_temp = []
    for t in time_list:
        # do inference
        data_out = recon_model.reconstruct_with_xgboost(
            t, coeff_list, reduction=None, num_modes=num_modes
        )

        out_temp.append(data_out.max())
    data_out = recon_model.reconstruct_with_xgboost(
        slider_time, coeff_list, reduction=None, num_modes=num_modes
    )
    del recon_model
    return np.c_[time_list, out_temp], data_out

@st.cache_data
def create_timeseries_plot(data):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Maximum temperature [K]")
    return fig

@st.cache_data
def get_parameters_from_config(config_name):
    """
    Automatically get uncertain physical parameters from uq-toolkit config
    file.
    Example config file is in the huggingface repo.
    Each uncertain param should have a "human-description" field to use
    as a label for the slider.
    We compute min and max values based on "distribution" params, assuming
    the distribution is normal

    Parameters
    ----------
    config_name : str
        /path/to/config.json
    """
    with open(config_name, "r") as f:
        config = hjson.load(f)

    # find all apps for UQ the uq-toolkit config file
    app_name_list = config["apps"].keys()

    name_list = []
    min_val_list = []
    max_val_list = []
    for fname in app_name_list:
        app_type = config["apps"][fname]["type"]
        # uq-toolkit supports "moose" and "json" for app configs,
        # we just use "moose" now
        if app_type == "moose":
            # shortcut for params in this specific app
            uq_config = config["apps"][fname]["uncertain-params"]
            for key_i in uq_config:
                for param_i in uq_config[key_i]:
                    # shortcut for the current parameter
                    uq_param_config = uq_config[key_i][param_i]
                    # check distribution is uniform,
                    # otherwise we cannot calc min and max values properly
                    if uq_param_config["distribution"]["name"] != "uniform":
                        raise NotImplementedError(
                            "we only implemented uniform priors so far"
                        )
                    # append these to list, then they can be used as kwargs
                    # for streamlit sliders
                    name_list.append(uq_param_config["human-description"])
                    min_val_list.append(
                        float(uq_param_config["distribution"]["loc"])
                    )
                    max_val_list.append(
                        float(
                            uq_param_config["distribution"]["loc"]
                            + uq_param_config["distribution"]["scale"]
                        )
                    )

    return name_list, min_val_list, max_val_list


def application_page():

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

    with st.sidebar:
        st.header("Surrogate Parameters")

        num_modes = st.slider(
            label="Number of PCA modes",
            min_value=1,
            max_value=4,
            value=4,
            step=1,
        )

        st.divider()
        st.header("Physical Parameters")

        name_list, min_val_list, max_val_list = get_parameters_from_config(
            "tmp_data/config.jsonc"
        )

        coefs_to_test = []
        slider_time = st.slider(
            label="Time [s]",
            min_value=5,
            max_value=60,
            value=60,
            step=5,
        )

        for name_i, min_val, max_val in zip(
            name_list, min_val_list, max_val_list
        ):
            mid_value = (max_val + min_val) / 2
            slider_val = st.slider(
                label=name_i,
                min_value=min_val,
                max_value=max_val,
                value=mid_value,
                step=mid_value / 100,
            )
            coefs_to_test.append(slider_val)

    tab_data, field_vals = generate_data(
        coeff_list=coefs_to_test,
        xgb_file=REGRESSION_MODEL_PATH,
        pod_file=SPATIAL_MODEL_PATH,
        num_modes=num_modes,
        slider_time=slider_time
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

        </div>
    """
        ),
        unsafe_allow_html=True,
    )

    st.header("Timeseries plot of HIVE pulse")
    st.pyplot(
        create_timeseries_plot(tab_data),
    )

    st.header("Temperature field at end of pulse")
    if platform.system() == "Darwin":
        old_write_field(
            "Temperature [K]",
            field_vals,
            "tmp_data/example_moose_output_temperature_out.e",
            "tmp_data/temp_field.vtk",
        )
    else:
        write_field(
            "Temperature [K]",
            field_vals,
            "tmp_data/example_moose_output_temperature_out.e",
            "tmp_data/temp_field.vtk",
        )
