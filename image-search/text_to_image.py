import json
import streamlit as st
import logging

from PIL import Image
import utils


def app():
    model, processor = utils.load_model(utils.MODEL_PATH, utils.BASELINE_MODEL)

    st.title("Retrieve Images given Text")
    query_text = st.text_input("Enter a text query:")
    query_type = st.radio("Select search type:",
        options=["text search", "vector search", "combined search"],
        index=0)

    if st.button("Search"):
        logging.info("text_to_image: {:s} ({:s})".format(query_text, query_type))
        st.text("returning results")

        if query_type == "text search":
            results = utils.do_text_search(query_text, utils.SEARCH_ENDPOINT_URL)
        elif query_type == "vector search":
            query_vec = utils.get_query_vec(query_text, model, processor)
            results = utils.do_vector_search(query_vec, utils.SEARCH_ENDPOINT_URL)
        else:
            query_vec = utils.get_query_vec(query_text, model, processor)
            results = utils.do_combined_search(query_text, query_vec, utils.SEARCH_ENDPOINT_URL)

        # st.text(json.dumps(results, indent=2))

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

