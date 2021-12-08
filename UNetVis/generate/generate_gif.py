from PIL import Image, ImageDraw
import os
from generate_feature_maps import get_model_names, get_layer_names
import glob
import imageio
def generate_one_dif(rootdir, patient, layer, slicename, channel, save_dir):
    # create save path
    patient_savedir = os.path.join(save_dir, patient)
    if not os.path.exists(patient_savedir):
        os.mkdir(patient_savedir)

    slice_savedir = os.path.join(patient_savedir, slicename)
    if not os.path.exists(slice_savedir):
        os.mkdir(slice_savedir)

    layer_savedir = os.path.join(slice_savedir, layer)
    if not os.path.exists(layer_savedir):
        os.mkdir(layer_savedir)

    channel_savedir = os.path.join(layer_savedir, str(channel))
    if not os.path.exists(channel_savedir):
        os.mkdir(channel_savedir)


    list_imgs = []
    for ith_model in range(2):
        current_path = os.path.join(rootdir, str(ith_model), patient, layer, slicename, str(channel) + '.png')
        input_image = Image.open(current_path)
        list_imgs.append(input_image)

    imageio.mimsave(os.path.join(channel_savedir, 'ev.gif', ), list_imgs, duration =1)

    return

def convert_all_to_gif(root_dir, png_dir, save_dir):
    # patients
    patient_dirpath = glob.glob(
        os.path.join(root_dir, "TCGA*")
    )
    list_patients = ['TCGA_CS_6665_20010817'] #[p[p.rfind("/") + 1 :] for p in patient_dirpath]

    list_models = [str(i) for i in range(100)]
    list_layers = ['upconv4'] #get_layer_names()

    list_slices = []

    for i_patient in list_patients:
        slicefilenames = []
        num_of_slices = len(os.listdir(os.path.join(root_dir, i_patient))) // 2

        for slice_id in range(num_of_slices):
            slicefilenames.append("{}_{}.tif".format(i_patient, slice_id + 1))

            for i_slice in slicefilenames:
                for i_layer in list_layers:
                        #num_of_channels = len(os.listdir(os.path.join(save_dir, '0', i_patient, i_layer, i_slice)))
                    channels = ['1'] #[str(ic+1) for ic in range(num_of_channels)]
                for i_channel in channels:
                    generate_one_dif(png_dir, i_patient, i_layer,i_slice, i_channel, save_dir)
    return

if __name__ == "__main__":
    root_dir = 'datasets/kaggle_3m' #'/playpen-ssd/jyn/brain-segmentation-pytorch-master/kaggle_3m/'
    png_dir = 'act_map' #''/playpen-ssd/jyn/infovis/act_map/'
    save_dir = '/playpen-ssd/jyn/infovis/gif/'
    convert_all_to_gif(root_dir, png_dir, save_dir)

