import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
import base64
import io






def vis_manifold():
    st.title("Latent space learned by 2D UNet")


    source = pd.read_csv('bottleneck/dimension_reduction.csv')
    a = alt.Chart(source).mark_circle(size=10).encode(
        x='X',
        y='Y',
        color='DICE',
        tooltip=['image',  'img_index', 'DICE']  # Must be a list for the image to render
    ).interactive()
    st.altair_chart(a, use_container_width=True)

    # select layers
    image_names = []
    for i in range(len(source)):
        image_names.append(str(i))
    selected_images = st.multiselect("Please select the images", image_names)

    # show mri and its mask
    if selected_images != "-" and selected_images:
        cols= st.columns(len(selected_images))

        for idx in range(len(selected_images)):
            current_idx = int(selected_images[idx])
            #st.text('Show Image, Gt Mask, and Segmentation Correspondingly below.')
            with cols[idx]:
                st.image(Image.open(source['img_path'].values[current_idx]), caption='Image')
                st.image(Image.open(source['gt_path'].values[current_idx]), caption='GT')
                st.image(Image.open(source['pred_path'].values[current_idx]), caption='Prediction')




def main():

    vis_manifold()

if __name__ == "__main__":
    main()
