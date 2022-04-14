import json
import numpy as np
import os
import requests
import streamlit as st
import torch

from transformers import CLIPModel, CLIPProcessor

BASELINE_MODEL = "openai/clip-vit-base-patch32"
MODEL_PATH = "/home/ubuntu/clip-model/clip-imageclef-run5-ckpt10"
APP_NAME = "clip-demo"
SCHEMA_NAME = "image"
GET_ENDPOINT_URL = "http://localhost:8080/document/v1/{:s}/{:s}/docid/{:d}"
SEARCH_ENDPOINT_URL = "http://localhost:8080/search/"


@st.cache(allow_output_mutation=True)
def load_model(model_path, baseline_model):
    model = CLIPModel.from_pretrained(model_path)
    # model = CLIPModel.from_pretrained(baseline_model)
    processor = CLIPProcessor.from_pretrained(baseline_model)
    return model, processor


def do_text_search(query_text, endpoint_url):
    headers = { "Content-type" : "application/json" }
    yql = """ select * from sources image where caption_text contains '{:s}'; """.format(
        query_text)
    params = {
        "yql": yql,
        "hits": 10,
        "ranking.profile": "caption-search"
    }
    resp = requests.post(endpoint_url, headers=headers, json=params)
    return resp.json()


def do_vector_search(query_vec, endpoint_url):
    headers = { "Content-Type" : "application/json" }
    params = {
        "yql": """select * from sources image where ([{"targetHits": 10}]nearestNeighbor(clip_vector, query_vector)); """,
        "hits": 10,
        "ranking.features.query(query_vector)": query_vec.tolist(),
        "ranking.profile": "image-search"
    }
    resp = requests.post(endpoint_url, headers=headers, json=params)
    return resp.json()


def do_combined_search(query_text, query_vec, endpoint_url):
    headers = { "Content-Type" : "application/json" }
    yql = """select * from sources image where ([{"targetHits": 10}]nearestNeighbor(clip_vector, query_vector)) OR caption_text contains '%s'; """ % (query_text)
    params = {
        "yql": yql,
        "hits": 10,
        "ranking.features.query(query_vector)": query_vec.tolist(),
        "ranking.profile": "combined-search"
    }
    data = json.dumps(params)
    resp = requests.post(endpoint_url, headers=headers, data=data)
    return resp.json()


def get_document_by_id(doc_id, endpoint_url):
    headers = { "Content-Type" : "application/json" }
    resp = requests.get(endpoint_url.format(APP_NAME, SCHEMA_NAME, doc_id), 
                        headers=headers)
    return resp.json()


def parse_image_vector_from_response(resp_json):
    emb = np.zeros((512), dtype=np.float32)
    for cell in resp_json["fields"]["clip_vector"]["cells"]:
        pos = int(cell["address"]["x"])
        val = cell["value"]
        emb[pos] = val
    return emb


def get_best_vector_from_text_query(query_text, endpoint_url, getdoc_url):
    text_results = do_text_search(query_text, endpoint_url)
    top_result_id = int(text_results["root"]["children"][0]["id"].split(":")[-1])
    docid_json = get_document_by_id(top_result_id, getdoc_url)
    emb = parse_image_vector_from_response(docid_json)
    return emb


def get_image_vector_and_caption_text_by_id(image_id, endpoint_url, getdoc_url):
    headers = { "Content-Type" : "application/json" }
    params = {
        "yql": """select documentid from sources image where image_id matches '%s'; """ % (image_id),
        "hits": 1,
    }
    resp = requests.post(endpoint_url, headers=headers, json=params)
    doc_id = int(resp.json()["root"]["children"][0]["id"].split(":")[-1])
    docid_json = get_document_by_id(doc_id, getdoc_url)
    image_vector = parse_image_vector_from_response(docid_json)
    caption_text = docid_json["fields"]["caption_text"]
    return image_vector, caption_text


def parse_results(result_json):
    metadata = {
        "num_results": result_json["root"]["fields"]["totalCount"]
    }
    data = []
    try:
        for child in result_json["root"]["children"]:
            data.append({
                "id": child["id"],
                "image_id": child["fields"]["image_id"],
                "image_path": child["fields"]["image_path"],
                "caption_text": child["fields"]["caption_text"],
                "relevance": child["relevance"]
            })
    except KeyError:
        pass
    return metadata, data


def get_query_vec(query_text, model, processor):
    inputs = processor([query_text],  padding=True, return_tensors="pt")
    with torch.no_grad():
        query_vec = model.get_text_features(**inputs)
    query_vec = query_vec.reshape(-1).numpy()
    query_vec /= np.linalg.norm(query_vec, 2)
    return query_vec
