import text_to_image
import image_to_image

import streamlit as st
import logging


logging.basicConfig(filename="streamlit_logs.txt")

PAGES = {
    "Test to Image Search": text_to_image,
    "Image to Image Search": image_to_image
}

st.sidebar.title("CLIP Image Search Demo")
st.sidebar.markdown("""
Demo to showcase image search capabilities of a CLIP transformer model
fine-tuned on the ImageCLEF 2017 Caption Prediction dataset.

CLIP is a transformer model from OpenAI that was trained on a large number
of image + text pairs from the Internet. CLIP learns a joint embedding for
images and their associated captions, such that images and their captions
are pushed closer together in the embedding space. The resulting CLIP model
can be used for text-to-image or image-to-image search, and performs well
on general images.

OOB performance on medical images is not as good, however. To remedy that, 
the CLIP model with medical images and captions from ImageCLEF, which resulted 
in significant performance improvements (as measured by MRR@k, for k=1, 3, 
5, 10, 20).

The CLIP model trained on ImageCLEF data was used to generate vectors for the
entire ImageCLEF dataset (training, validation, and unseen test images) and 
loaded into a Vespa search index, which provides the Approximate Nearest 
Neighbors vector search functionality for this demo.
""")

selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
