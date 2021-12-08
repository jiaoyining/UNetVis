"""Visualize activation maps for each channel."""
import base64
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
import imageio

def get_path_feature_map( datasedir, patient, selected_slice, rootdir='gif'):
    #st.text('Showing the ' + str(selected_slice) + ' slice now!')

    layer_path = dict()
    layer_names = [
        'input', 'encoder1', 'encoder2', 'encoder3', 'encoder4',
        'a', 'pool1', 'pool2', 'pool3', 'pool4', 'bottleneck',
        'a', 'upconv1', 'upconv2', 'upconv3', 'upconv4',
        'conv', 'decoder1', 'decoder2', 'decoder3', 'decoder4', ]
    a = np.zeros((256, 256))
    b = np.zeros((256, 256))
    pad_path=os.path.join(rootdir, 'pad.gif')
    imageio.mimsave(pad_path,[a, b], duration =0.5)

    for i_layername in layer_names:

        if i_layername != 'a':
            if i_layername == 'input':
                mri_filename = "{}_{}.tif".format(patient, selected_slice )
                image_path = os.path.join(datasedir, patient, mri_filename)
                a = Image.open(image_path)
                #st.text(image_path)
                imageio.mimsave('out.gif', [a, a], duration=0.5)
                layer_path[i_layername] = 'out.gif'
            else:
                mri_filename = "{}_{}.tif".format(patient, selected_slice)
                i_layer_path = os.path.join(rootdir, patient,mri_filename, i_layername,  '1', 'ev.gif')
                layer_path[i_layername] = i_layer_path
        else:
            layer_path[i_layername] = pad_path
    return layer_path


def gen_act_lgg_mri(model, layer, patient, slice_id, channels):
    # register hook
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    prefix = "model." + layer
    eval(prefix).register_forward_hook(get_activation(layer))

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

    # generate image
    if len(channels) < 3:
        fig_col = 3
    else:
        fig_col = len(channels)

    fig, axs = plt.subplots(3, fig_col, figsize=(fig_col, 2), sharex=True, sharey=True)
    remove_all_spines(axs)
    # pred mask
    pred_mask = output[0][0].detach().cpu().numpy()
    pred_mask = np.round(pred_mask)
    axs[0, 0].imshow(input_image)
    mask_filename = "{}_{}_mask.tif".format(patient, slice_id)
    mask_path = os.path.join(patient_dirpath, mask_filename)
    mask = imread(mask_path)
    axs[0, 1].imshow(mask, cmap="gray")
    axs[0, 2].imshow(pred_mask, cmap="gray")
    # activiation map
    for i, channel_id in enumerate(channels):
        channel_id = int(channel_id)
        A = activation[layer][0][channel_id].cpu().numpy()
        S = resize(A, (256, 256))
        axs[1, i].imshow(S, cmap="gray")
        axs[1, i].set_xlabel(channel_id)

    axs[1, 0].set_ylabel("act ({})".format(A.shape[0]))

    plt.xticks([])
    plt.yticks([])
    return fig


def vis_evolution_for_a_featmap():
    # load patients
    patient_dirpath = glob.glob(
        os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", "TCGA*")
    )
    patient_names = [p[p.rfind("/") + 1 :] for p in patient_dirpath]
    patient_names.insert(0, "-")
    selected_patient = st.selectbox("Please select the patient", ['TCGA_CS_6667_20011105'])

    # show slice and its mask
    selected_slice_id = None
    if selected_patient != "-":
        fig, slice_num = gen_patient_lgg_mri(selected_patient)
        st.pyplot(fig, bbox_inches="tight", use_column_width=True)

        # select specific slice
        selected_slice_id = st.slider(
            "Please select the slice", min_value=1, max_value=slice_num
        )

    '''
    selected_layers = None
    if selected_patient != "-":
        # select layers
        layer_names = []
        model = UNet(in_channels=3, out_channels=1, init_features=32)
        for name, param in model.named_children():
            layer_names.append(name)
        selected_layers = st.multiselect("Please select the layers", layer_names)
    '''
    # show buttons
    show = None
    if selected_slice_id:
        show = st.button("show")

    selected_layers = [
        ['input', 'encoder1', 'encoder2', 'encoder3', 'encoder4',],
        ['a', 'pool1', 'pool2', 'pool3', 'pool4', 'bottleneck'],
        ['a', 'upconv1', 'upconv2', 'upconv3', 'upconv4',],
        ['conv', 'decoder1', 'decoder2', 'decoder3', 'decoder4', ]]
    layer_type = ['encoder', 'pool', 'upconv', 'decoder']
    cols= st.columns(4)
    dict_gif = dict()
    if show:
        datasedir = 'datasets/lgg-mri-segmentation/kaggle_3m/'
        dict_gif = get_path_feature_map(datasedir, selected_patient, selected_slice_id, rootdir='gif')
        for i_layer_type in range(4):
            with cols[i_layer_type]:
                st.header(layer_type[i_layer_type])
                current_layer_names = selected_layers[i_layer_type]

                for i_current_layer in current_layer_names:
                    current_gif_path = dict_gif[i_current_layer]

                    file_ = open(current_gif_path, "rb")
                    contents = file_.read()
                    data_url = base64.b64encode(contents).decode("utf-8")
                    file_.close()

                    st.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" alt="ev gif" width="200" height="200">',
                        unsafe_allow_html=True,
                    )




def vis_evolution():


    # load dataset
    selected_dataset = None
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
        vis_evolution_for_a_featmap()


def main():
    st.title("Evolution of Activation Maps")
    gif_dir = ''
    vis_evolution()


if __name__ == "__main__":
    main()
