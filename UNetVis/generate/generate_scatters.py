import pickle5 as pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
from PIL import Image
import io
import base64
def np_image_to_base64(path):
    im_matrix = np.array(Image.open(path))
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def generate_manifold_pickle_file(dataset_path='/Users/jyn/jyn/course/InfoVis/UNetVis/datasets/lgg-mri-segmentation/kaggle_3m/',
                                  prediction_path='../predictions/'):
    file = open("../bottleneck/feature_list.pickle",'rb')
    object_file = pickle.load(file)

    list_features = []
    list_names = []
    list_mask = []
    for i in range(len(object_file)):
        list_features.append(object_file[i]['feature'])
        list_names.append(object_file[i]['name'])
        list_mask.append(object_file[i]['mask'])
    arr_features = np.array(list_features)


    X_embedded = TSNE(n_components=2, learning_rate='auto',  init='random').fit_transform(arr_features)


    saved_file = '../bottleneck/dimension_reduction.csv'
    list_scatters = []
    for i in range(len(list_names)):
        current_img_path = os.path.join(dataset_path, list_names[i])
        current_gt_path = os.path.join(dataset_path, list_mask[i])
        current_pred_path = os.path.join(prediction_path, object_file[i]['pred_img'])
        list_scatters.append({'image': np_image_to_base64(current_img_path),
                         'X': X_embedded[i, 0],
                         'Y': X_embedded[i, 1],
                         'img_path': current_img_path,
                         'name': object_file[i]['name'],
                         'pred_path': current_pred_path,
                          'gt_path': current_gt_path,
                         'DICE': object_file[i]['DICE'],
                          'img_index': i })

    pd.DataFrame.from_records(list_scatters).to_csv(saved_file)


def main():
    dataset_path = 'https://www.dropbox.com/sh/uqc6ot1uq7jm3t0/AABhvVon8yGXiBb0itoZ9uvNa?dl=0' #'/Users/jyn/jyn/course/InfoVis/UNetVis/datasets/lgg-mri-segmentation/kaggle_3m/'
    prediction_path = 'predictions/'

    generate_manifold_pickle_file(dataset_path, prediction_path)


if __name__ == "__main__":
    device = 'cpu'
    main()