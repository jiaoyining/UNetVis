"""Visualize activation maps for each channel."""

from skimage.transform import resize
from PIL import Image
import numpy as np
import streamlit as st

try:
    from model import *
    from util import *
except ImportError:
    from .model import *
    from .util import *


def gen_act_lgg_mri(model, layers, patient, slice_id):
    # register hook
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    for layer_name in layers:
        prefix = "model." + layer_name
        eval(prefix).register_forward_hook(get_activation(layer_name))


    # model inference
    patient_dirpath = os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", patient)

    slice_filename = "{}_{}.tif".format(patient, slice_id)
    slice_path = os.path.join(patient_dirpath, slice_filename)
    input_image = Image.open(slice_path)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    input_tensor = torch.tensor((input_image - m) / s, dtype=torch.float)
    input_tensor = input_tensor.permute([2, 0, 1])

    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    model = model.to(device)
    model.eval()
    output = model(input_batch)

    plt.close()

    cols = st.columns(1+len(layers))


    with cols[0]:
        st.subheader('Basics')
        patient_dirpath = os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", patient)
        # mri
        mri_filename = "{}_{}.tif".format(patient, slice_id)
        mri_path = os.path.join(patient_dirpath, mri_filename)
        mri = Image.open(mri_path)
        plt.imshow(mri)
        plt.title('Image')
        plt.axis('off')
        st.pyplot(plt, bbox_inches="tight")
        plt.close()

        patient_dirpath = os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", patient)
        # gt
        gt_filename = "{}_{}_{}.tif".format(patient, slice_id, 'mask')
        gt_path = os.path.join(patient_dirpath, gt_filename)
        gt = Image.open(gt_path)
        plt.imshow(gt, cmap='gray')
        plt.title('GT')
        plt.axis('off')
        st.pyplot(plt, bbox_inches="tight")
        plt.close()

        #st.header("Prediction")
        # pred mask
        pred_mask = output[0,0].detach().cpu().numpy()
        pred_mask = np.round(pred_mask)
        plt.imshow(pred_mask, cmap='gray')
        plt.title('pred')
        plt.axis('off')
        st.pyplot(plt, bbox_inches="tight")
        plt.close()


    for idx_layer in range(len(layers)):
        with cols[idx_layer+1]:
            st.subheader(layers[idx_layer])
            # activiation map
            A = activation[layers[idx_layer]][0].cpu().numpy()
            channels = np.arange(A.shape[0])
            for i_channel in channels:
                S = resize(A[i_channel], (256, 256))
                plt.imshow(S, cmap='hot')
                plt.title(layers[idx_layer])
                plt.axis('off')
                st.pyplot(plt)
                plt.close()


    return 0


def vis_lgg_mri(models):
    # load patients
    patient_dirpath = glob.glob(
        os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", "TCGA*")
    )
    patient_names = [p[p.rfind("/") + 1 :] for p in patient_dirpath]
    patient_names.insert(0, "-")
    selected_patient = st.selectbox("Please select the patient", patient_names)

    # show slice and its mask
    selected_slice_id = None
    if selected_patient != "-":
        fig, slice_num = gen_patient_lgg_mri(selected_patient)
        st.pyplot(fig, bbox_inches="tight", use_column_width=True)

        # select specific slice
        selected_slice_id = st.slider(
            "Please select the slice", min_value=1, max_value=slice_num
        )


    # show mri and its mask
    selected_layers = None
    if selected_patient != "-":
        # select layers
        layer_names = []
        model = UNet(in_channels=3, out_channels=1, init_features=32)
        for name, param in model.named_children():
            layer_names.append(name)
        selected_layers = st.multiselect("Please select the layers", layer_names)

    if selected_layers:
        # register hook
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        for layer_name in selected_layers:
            prefix = "model." + layer_name
            eval(prefix).register_forward_hook(get_activation(layer_name))

        model.eval()


    # show buttons
    show = None
    if selected_layers:
        show = st.button("show")

    # show act maps
    if show:
        st.write(selected_layers)
        for name, model in models.items():
            st.text(name)
            with st.spinner("generating activation maps..."):
                fig = gen_act_lgg_mri(
                    model,
                    selected_layers,
                    selected_patient,
                    selected_slice_id,
                )
                #st.pyplot(fig, bbox_inches="tight")


def vis_activation_maps_channels():
    # filter checkpoint
    ckpt_paths = glob.glob(os.path.join(cur_dir, "checkpoints", "*.pt"))
    ckpt_names = [p[p.rfind("/") + 1 :] for p in ckpt_paths]
    ckpt_names.insert(0, "-")
    selected_ckpts = st.multiselect("Please select the checkpoints", ckpt_names)

    # load model
    models = {}
    if selected_ckpts:
        for name in selected_ckpts:
            with st.spinner("loading {}...".format(name)):
                path = os.path.join(cur_dir, "checkpoints", name)
                model, missing_keys, unexpected_keys = load_model(path)
                if missing_keys:
                    st.error("missing keys for {} \n\n {}".format(name, missing_keys))
                    return -1
                else:
                    models[name] = model

    # load dataset
    selected_dataset = None
    if models:
        # select dataset
        avaiable_dataset_paths = glob.glob(os.path.join(cur_dir, "datasets", "*/"))
        avaiable_dataset_names = [
            os.path.basename(path[:-1]) for path in avaiable_dataset_paths
        ]

        avaiable_dataset_names.insert(0, "-")
        selected_dataset = st.selectbox(
            "Please select the dataset", avaiable_dataset_names
        )

    # visualization
    if selected_dataset == "lgg-mri-segmentation":
        vis_lgg_mri(models)


def main():
    st.title("2D Unet Activation Maps (Channel)")
    vis_activation_maps_channels()


if __name__ == "__main__":
    main()



