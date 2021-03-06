"""Main entrance of the project."""

import streamlit as st

# import each visualization here
# import each visualization here
from vis import vis_manifold
from vis import vis_diagram
from vis import vis_evolution
from vis import vis_act_map
from vis import vis_filter_image
from vis import vis_act_map_channel
from vis import vis_filter_stats

from vis.util import *


def show_default_page():
    st.header('UNetVis')
    st.subheader("Task:")
    st.markdown(
        "Segmentation is the process of partitioning an image into different meaningful segments. In medical imaging, these segments often correspond to different tissue classes, organs, pathologies, or other biologically relevant structures."
    )
    st.image(os.path.join(cur_dir, "assets", "example.png"), use_column_width=True)

    st.subheader("UNet:")
    st.markdown(
        "The u-net is convolutional network architecture for fast and precise segmentation of images."
    )
    st.image(os.path.join(cur_dir, "assets", "model.png"), use_column_width=True)

    st.subheader("LGG Segmentation Dataset:")
    st.markdown(
        "This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks. The images were obtained from The Cancer Imaging Archive (TCIA). They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available."
    )
    st.image(
        os.path.join(cur_dir, "assets", "lgg_mri_dataset.png"), use_column_width=True
    )


def main():
    st.sidebar.title("UNetVis")

    # select visualizations
    vis_type = st.sidebar.selectbox(
        "What kind of visualization you want?",
        [
            "-",
            "Digram",
            "Filter images",
            "Filter stats",
            "Activation maps",
            "Activation maps (Channel)",
            "Evolution of Activation Maps",
            "Manifold"
        ],
    )

    if vis_type == "-":
        show_default_page()
    elif vis_type == "Digram":
        vis_diagram()
    elif vis_type == "Filter images":
        vis_filter_image()
    elif vis_type == "Filter stats":
        vis_filter_stats()
    elif vis_type == "Activation maps":
        vis_act_map()
    elif vis_type == "Activation maps (Channel)":
        vis_act_map_channel()
    elif vis_type == "Manifold":
        vis_manifold()
    elif vis_type == "Evolution of Activation Maps":
        vis_evolution()


if __name__ == "__main__":
    main()
