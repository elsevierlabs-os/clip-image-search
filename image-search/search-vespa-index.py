import argparse
import json
import numpy as np
import requests

# from requests.sessions import dispatch_hook

APP_NAME = "clip-demo"
SCHEMA_NAME = "image"
GET_ENDPOINT_URL = "http://localhost:8080/document/v1/{:s}/{:s}/docid/{:d}"
SEARCH_ENDPOINT_URL = "http://localhost:8080/search/"


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


def get_image_vector_by_id(image_id, endpoint_url, getdoc_url):
    headers = { "Content-Type" : "application/json" }
    params = {
        "yql": """select documentid from sources image where image_id matches '%s'; """ % (image_id),
        "hits": 1,
    }
    resp = requests.post(endpoint_url, headers=headers, json=params)
    doc_id = int(resp.json()["root"]["children"][0]["id"].split(":")[-1])
    docid_json = get_document_by_id(doc_id, getdoc_url)
    image_vector = parse_image_vector_from_response(docid_json)
    return image_vector


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


def pad_truncate(text, length):
    pad = " " * length
    text = (text + pad)[0:length]
    return text


def print_results(metadata, data):
    print("## Top 10 of {:d} results".format(metadata["num_results"]))
    for row in data:
        print("{:20s} | {:50s} | {:.3f}".format(
            pad_truncate(row["image_id"], 20),
            pad_truncate(row["caption_text"], 50),
            row["relevance"]
        ))


################################## main ##################################

parser = argparse.ArgumentParser()
parser.add_argument("--query_type", "-t",
                    choices=["text", "vector", "combined"],
                    default="text",
                    help="specify type of query to use")
parser.add_argument("--query", "-q", 
                    default="X-rays",
                    help="text of query")
args = parser.parse_args()

query_type = args.query_type
query_text = args.query

search_results = None
if query_type == "text":
    search_results = do_text_search(query_text, SEARCH_ENDPOINT_URL)
elif query_type == "vector":
    query_vector = get_best_vector_from_text_query(
        query_text, SEARCH_ENDPOINT_URL, GET_ENDPOINT_URL)
    search_results = do_vector_search(query_vector, SEARCH_ENDPOINT_URL)
else:
    query_vector = get_best_vector_from_text_query(
        query_text, SEARCH_ENDPOINT_URL, GET_ENDPOINT_URL)
    search_results = do_combined_search(
        query_text, query_vector, SEARCH_ENDPOINT_URL)

print("--- {:s} search results ---".format(query_type))
metadata, data = parse_results(search_results)
print_results(metadata, data)

# image_vector = get_image_vector_by_id('ORT-1745-3674-80-548-g002', 
#     SEARCH_ENDPOINT_URL, GET_ENDPOINT_URL)
# print(image_vector)
