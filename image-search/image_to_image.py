import json
import logging
import streamlit as st

from PIL import Image
import utils


def app():
    model, processor = utils.load_model(utils.MODEL_PATH, utils.BASELINE_MODEL)

    st.title("Retrieve Images given an Image")
    image_id = st.text_input("Enter Image-ID:")
    query_type = st.radio("Select search type:",
        options=["image-only", "image+text"],
        index=0)

    if st.button("Search"):
        logging.info("image_to_image: {:s} ({:s})".format(image_id, query_type))
        st.text("returning results")
        image_vec, caption_text = utils.get_image_vector_and_caption_text_by_id(
            image_id, utils.SEARCH_ENDPOINT_URL, utils.GET_ENDPOINT_URL)
        if query_type == "image-only":
            results = utils.do_vector_search(image_vec, utils.SEARCH_ENDPOINT_URL)
        else:
            results = utils.do_combined_search(caption_text, image_vec, utils.SEARCH_ENDPOINT_URL)

        metadata, data = utils.parse_results(results)
        st.markdown("## Results 1-{:d} from {:d} matches".format(
            len(data), metadata["num_results"]))
        for rid, row in enumerate(data):
            image_id = row["image_id"]
            image_path = row["image_path"]
            image = Image.open(image_path).convert("RGB")
            caption = row["caption_text"]
            relevance = row["relevance"]
            col1, col2, col3 = st.columns([2, 10, 10])
            col1.markdown("{:d}.".format(rid + 1))
            col2.image(image)
            col3.markdown("""
            * **Image-ID**:      {:s}
            * **Caption**:   {:s}
            * **Relevance**: {:.3f}
            """.format(image_id, caption, relevance))
            st.markdown("---")

