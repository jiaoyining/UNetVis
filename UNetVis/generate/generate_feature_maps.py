"""Visualize activation maps for each channel."""
import glob

import torch
from skimage.transform import resize
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import os

from model import UNet
from util import load_model
from tqdm import trange
# models
def generate_all(model_dir, root_dir, save_dir):

    list_models = get_model_names(model_dir)
    for idx in trange(len(list_models)):
        i_model_path = list_models[idx]
        model, missing_keys, unexpected_keys = load_model(i_model_path)
        save_dir_model = os.path.join(save_dir, str(idx))
        if not os.path.exists(save_dir_model):
            os.mkdir((save_dir_model))
        generate_act_maps_for_model(model, root_dir, save_dir_model)
    return 0


def generate_act_maps_for_model(model, root_dir, save_dir):

    # patients
    patient_dirpath = glob.glob(
        os.path.join(root_dir, "TCGA*")
    )
    patient_names = [[p[p.rfind("/") + 1 :] for p in patient_dirpath][0]]
    #patient_names.insert(0, "-")

    for idx in trange(len(patient_names)):
        i_patient = patient_names[idx]
        save_dir_patient = os.path.join(save_dir, i_patient)
        if not os.path.exists(save_dir_patient):
            os.mkdir(save_dir_patient)
        generate_act_maps_for_patient(model, root_dir, i_patient, save_dir_patient)

    return 0

def generate_act_maps_for_patient(model, root_dir, patient, save_dir):
    layer_names = get_layer_names()
    #model, missing_keys, unexpected_keys = load_model(model_path)
    slices = []
    slicefilenames = []
    patient_path = os.path.join(root_dir, patient)
    num_of_slices = len(os.listdir(patient_path)) // 2

    for slice_id in trange(num_of_slices):
        slicefilenames.append("{}_{}.tif".format(patient, slice_id+1))

    #slice_path = os.path.join(patient_path, slice_filename)

    for i_layer in layer_names:
        save_dir_patient_layer = os.path.join(save_dir, i_layer)
        if not os.path.exists(save_dir_patient_layer):
            os.mkdir(save_dir_patient_layer)
        for i_slice in slicefilenames:
            save_dir_patient_layer_slice = os.path.join(save_dir_patient_layer, i_slice)
            if not os.path.exists(save_dir_patient_layer_slice):
                os.mkdir(save_dir_patient_layer_slice)
            i_slice_path = os.path.join(patient_path, i_slice)
            gen_individual_act_maps(model, i_layer, i_slice_path, save_dir_patient_layer_slice)


def gen_individual_act_maps(model, layer, slice_path, save_path):
    # register hook
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    prefix = "model." + layer
    eval(prefix).register_forward_hook(get_activation(layer))

    input_image = Image.open(slice_path)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    input_tensor = torch.tensor((input_image - m) / s, dtype=torch.float)
    input_tensor = input_tensor.permute([2, 0, 1])

    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    model = model.to(device)
    model.eval()
    output = model(input_batch)

    num_act_maps = activation[layer].shape[1]
    channels = [str(x) for x in list(range(num_act_maps))]
    '''
    # activiation map
    for idx in trange(len(channels)):
        channel_id = channels[idx]
        channel_id = int(channel_id)
        A = activation[layer][0][channel_id].cpu().numpy()
        #A = np.average(A, axis=0)
        S = resize(A, (256, 256))
        plt.imshow(S,  cmap='hot')
        plt.xticks([])
        plt.yticks([])
        filename = os.path.join(save_path, str(idx+1) + '.png')
        plt.axis('off')

        plt.savefig(filename , bbox_inches='tight', pad_inches=0.0)
    '''
    # activiation map

    A = activation[layer][0].cpu().numpy()
    A = np.average(A, axis=0)
    S = resize(A, (256, 256))
    plt.imshow(S,  cmap='hot')
    plt.xticks([])
    plt.yticks([])
    filename = os.path.join(save_path, str(1) + '.png')
    plt.axis('off')

    plt.savefig(filename , bbox_inches='tight', pad_inches=0.0)
    return 0

def get_model_names(model_addr):
    list_model = []
    for i in range(10):
        list_model.append(os.path.join(model_addr, str(i*10) + '.pt'))
    return list_model

def get_layer_names():
    layer_names = []
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    for name, param in model.named_children():
        layer_names.append(name)
    return layer_names



def main():
    model_dir = '/playpen-ssd/jyn/brain-segmentation-pytorch-master/weights/'
    root_dir = '/playpen-ssd/jyn/brain-segmentation-pytorch-master/kaggle_3m/'
    save_dir = '/playpen-ssd/jyn/infovis/act_map/'

    generate_all(model_dir, root_dir, save_dir)


if __name__ == "__main__":
    device = 'cuda:0'
    main()
