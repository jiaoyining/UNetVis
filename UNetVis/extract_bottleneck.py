"""Visualize activation maps for each channel."""
import glob

import torch
from PIL import Image
import numpy as np

import os

from vis import load_model
from tqdm import trange
import pickle
import sys



def dice_score(im1, im2):

    im1 = np.asarray(im1).astype('bool')
    im2 = np.asarray(im2).astype('bool')

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    if (im1.sum() + im2.sum()) == 0:
        return 1
    else:
        return 2. * intersection.sum() / (im1.sum() + im2.sum())


# models
def generate_all(model_dir, dataset_dir, save_dir='bottleneck/feature_list.pickle'):


    patient_dirpath = glob.glob(
        os.path.join(dataset_dir, "TCGA*")
    )
    list_patient = [p[p.rfind("/") + 1 :] for p in patient_dirpath]

    list_feature = []


    for idx in trange(len(list_patient)):
        i_patient = list_patient[idx]
        #model, missing_keys, unexpected_keys = load_model(i_model_path)
        '''
        save_dir_patient = os.path.join(save_dir, i_patient)
        if not os.path.exists(save_dir_patient):
            os.mkdir((save_dir_patient))
        '''
        list_feature = generate_act_maps_for_patient(model_dir, dataset_dir, i_patient, list_feature)

    file = open(save_dir, 'wb')
    pickle.dump(list_feature, file)
    file.close()

    return 0


def generate_act_maps_for_patient(model_dir, dataset_dir, patient,list_feature):

    # patients
    list_model = get_model_names(model_dir)

    for idx in trange(len(list_model)):
        model, missing_keys, unexpected_keys = load_model(list_model[idx])
        '''
        save_dir_model = os.path.join(save_dir, str(idx))
        if not os.path.exists(save_dir_model):
            os.mkdir(save_dir_model)
        '''
        list_feature = generate_act_maps_for_model(model, dataset_dir, patient, list_feature)

    return list_feature

def generate_act_maps_for_model(model, dataset_dir, patient, list_feature):
    layer_names = get_layer_names()
    #model, missing_keys, unexpected_keys = load_model(model_path)
    slices = []
    slicefilenames = []
    patient_path = os.path.join(dataset_dir, patient)
    num_of_slices = len(os.listdir(patient_path)) // 2

    maskfilenames = []
    for slice_id in trange(num_of_slices):
        slicefilenames.append("{}_{}.tif".format(patient, slice_id+1))
        maskfilenames.append("{}_{}_mask.tif".format(patient, slice_id+1))
    #slice_path = os.path.join(patient_path, slice_filename)

    for i_layer in layer_names:
        '''
        save_dir_patient_layer = os.path.join(save_dir, i_layer)
        if not os.path.exists(save_dir_patient_layer):
            os.mkdir(save_dir_patient_layer)
            '''
        for i_slice in range(len(slicefilenames)):
            '''
            save_dir_patient_layer_slice = os.path.join(save_dir_patient_layer, i_slice)
            if not os.path.exists(save_dir_patient_layer_slice):
                os.mkdir(save_dir_patient_layer_slice)
                '''
            i_slice_path = os.path.join(patient_path, slicefilenames[i_slice])
            current_image_name = slicefilenames[i_slice]

            current_mask_name = maskfilenames[i_slice]
            current_mask_path = os.path.join(patient_path,  maskfilenames[i_slice])
            list_feature = gen_individual_act_maps(model, i_layer,
                                                   i_slice_path,
                                                   patient,
                                                   current_image_name,
                                                   current_mask_name,
                                                   current_mask_path,
                                                   list_feature)

    return list_feature

def gen_individual_act_maps(model, layer, slice_path, patient, current_image_name,  current_mask_name, current_mask_path, list_feature):
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
    input_tensor = torch.tensor((input_image - m) / (s+1e-5), dtype=torch.float)
    input_tensor = input_tensor.permute([2, 0, 1])

    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    model = model.to(device)
    model.eval()
    output = model(input_batch)
    current_pred = output[0, 0].detach().cpu().numpy()>0.5
    if not os.path.exists('predictions'):
        os.mkdir('predictions')
    Image.fromarray(current_pred.astype('uint8')*255).save(os.path.join('predictions', current_mask_name))

    current_mask = np.array(Image.open(current_mask_path))>0.5
    #print( np.array(Image.open(current_mask_name)).shape)
    current_dicescore = dice_score(current_pred.astype('float'), current_mask.astype('float'))


    num_act_maps = activation[layer].shape[1]
    channels = [str(x) for x in list(range(num_act_maps))]

    # activiation map

    A = activation[layer][0].cpu().numpy()
    #print(A.shape)
    A = np.average(A, axis=0)

    current_dict = {'name': patient + '/' + current_image_name,
                    'pred_img': current_mask_name,
                    'feature': A.flatten(),
                    'mask': patient + '/' + current_mask_name,
                    'DICE': current_dicescore}
    list_feature.append(current_dict)
    '''
    filename = os.path.join(save_path, str(1) + '.pkl')
    file = open('pickle_example.pickle', 'wb')
    pickle.dump(a_dict, file)
    file.close()
    '''

    return list_feature

def get_model_names(model_addr):
    list_model = ['checkpoints/unet-e012d006.pt']
    return list_model

def get_layer_names():
    layer_names = ['bottleneck']
    return layer_names



def main():
    model_dir = 'checkpoints'  #'/playpen-ssd/jyn/brain-segmentation-pytorch-master/weights/'
    dataset_dir = 'datasets/lgg-mri-segmentation/kaggle_3m'  #'/playpen-ssd/jyn/brain-segmentation-pytorch-master/kaggle_3m/'

    generate_all(model_dir, dataset_dir)


if __name__ == "__main__":
    device = 'cpu'
    main()
