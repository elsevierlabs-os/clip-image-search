# clip-image-search

Fine-tuning OpenAI CLIP Model for Image Search on medical images

* [Motivation](#motivation)
* [Applications](#applications)
* [References / Previous Work](#references--previous-work)
* [Fine Tuning](#fine-tuning)
  * [Environment](#environment)
  * [Data Preparation](#data-preparation)
  * [Training Hyperparameters](#training-hyperparameters)
  * [Outputs](#outputs)
  * [Evaluation](#evaluation)
* [Image Search](#image-search)
  * [Environment](#environment-1)
  * [Vespa](#vespa)
  * [Streamlit](#streamlit)
  * [Automatic Startup and Shutdown](#automatic-startup-and-shutdown)
    * [Vespa](#vespa-1)
    * [Streamlit](#streamlit-1)
    * [Ngnix](#nginx)


## Motivation

* Model based image search (i.e. using machine learning based models trained on image similarity rather than traditional Lucene based search on captions)
* Unsupervised or self supervised, because
  * labeling is expensive
  * we have lots of captioned images in our collection, so many (image, caption) pairs available "in the wild"
* This project is **not** about caption prediction, rather it exploring the feasibility of text-to-image image search, whereby the user enters a text string to bring up the most appropriate images for the text string.

## Applications

* Image search for internal users to get editorial efficiencies
* Image search for customers
* Captioning of new images, concepts in new images
* Image decomposition and caption assignment - an enabling technology for machine learning

## References / Previous Work

* [Contrastive Learning of Medical Visual Representations from Paired Images and Text](https://arxiv.org/abs/2010.00747) (Zhang et al, 2020)
  * learns visual representation of medical images and associated text using contrastive learning
  * uses different encoders for different specialties, eg, chest image encoder, bone image encoder
* [Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al, 2021)
  * led to the OpenAI CLIP model on which this project is based
  * [CLIP: Connecting Text and Images](https://openai.com/blog/clip/) -- blog post provides high level intro
  * [CLIP Hugging Face Model](https://huggingface.co/transformers/model_doc/clip.html)
  * [CLIP-rsicd project](https://github.com/arampacha/CLIP-rsicd)
    * same idea applied to satellite images
    * done with external team as part of Hugging Face Flax/JAX community week

## Fine Tuning

### Environment

* AWS EC2 p2.xlarge 
  * 4 CPUs
  * 1 GPU (Tesla K80)
  * 64 GB RAM
  * 300 GB disk
* Ubuntu 18.04 (bionic)
* [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/prodview-x5nivojpquy6y)
* Conda Environment: pytorch_latest_p37
  * Python 3.7.10
  * Pytorch 1.8.1+cu111
* Additional packages 
  * Transformers 4.10.0
  * see [requirements.txt](requirements.txt)

### Data Preparation

* Data for task originally provided via the [ImageCLEF 2017 Caption Prediction task](https://www.imageclef.org/2017/caption).
* Input data located at [s3://els-corpora/ImageCLEF/ImageCLEF2017/Caption/CaptionPrediction](https://s3.console.aws.amazon.com/s3/buckets/els-corpora?region=us-east-1&prefix=ImageCLEF/ImageCLEF2017/Caption/CaptionPrediction/&showversions=false).
* Download using following command:
```
$ mkdir ImageCLEF2017-CaptionPrediction
$ cd ImageCLEF2017-CaptionPrediction
$ aws s3 cp --recursive s3://els-corpora/ImageCLEF/ImageCLEF2017/Caption/CaptionPrediction .
```
* Dataset contains following splits
  * training: 164,614 images + captions
  * validtion: 10,000 images + captions
  * test: 10,000 images (no captions)
* We need 3 splits -- training, validation, and test.
  * We cannot use test for training or evaluation, so we discard it
  * validation data becomes test data
  * training data is split 90:10 into new training and validation
* End count is as follows:
  * training: 148,153 images + captions
  * validation: 16,461 images + captions
  * test: 10,000 images + captions

### Training Hyperparameters

* Best model hyperparameters
  * batch size: 64
  * optimizer: ADAM
  * learning rate: 5e-5
  * number of epochs: 10
  * number of training samples: 50,000
* We note that loss continues to drop so it is likely that further training or with larger amounts of data will increase performance. However, the flattening of the validation curve shows that we are in an area of diminishing returns.
  * <img src="images/lossplot_5.png"/>
* We considered doing image and text augmentation but dropped the idea since training set size is quite large (148k+ images+captions) and we achieve regularization through random sampling a subset of this dataset.


### Outputs

* Best model (run 5, ckpt 10) - [s3://els-ats/sujit/clip-imageclef/models/clip-imageclef-run5-ckpt10](https://s3.console.aws.amazon.com/s3/buckets/els-ats?region=us-east-1&prefix=sujit/clip-imageclef/models/&showversions=false)
* Vectors generated from best model
  * for test set images - [s3://els-ats/sujit/clip-imageclef/vectors/vectors-5.10.tsv](https://s3.console.aws.amazon.com/s3/object/els-ats?region=us-east-1&prefix=sujit/clip-imageclef/vectors/vectors-5.10.tsv)
  * for unseen (original test set) images - [s3://els-ats/sujit/clip-imageclef/vectors/vectors-unseen-5.10.tsv](https://s3.console.aws.amazon.com/s3/object/els-ats?region=us-east-1&prefix=sujit/clip-imageclef/vectors/vectors-unseen-5.10.tsv)
  * for validation images - [s3://els-ats/sujit/clip-imageclef/vectors/vectors-val-5.10.tsv](https://s3.console.aws.amazon.com/s3/object/els-ats?region=us-east-1&prefix=sujit/clip-imageclef/vectors/vectors-val-5.10.tsv)
  * for training images - [s3://els-ats/sujit/clip-imageclef/vectors/vectors-train-5.10.tsv](https://s3.console.aws.amazon.com/s3/object/els-ats?region=us-east-1&prefix=sujit/clip-imageclef/vectors/vectors-train-5.10.tsv)


### Evaluation

* We feed in batches of (caption-image) pairs
* Evaluation metrics based on the intuition illustrated by heatmap, i.e. labels for each batch are along the diagonal
  * <img src="images/eval_heatmap.png"/>
* We compute MRR@k (Mean Reciprocal Rank) for k=1, 3, 5, 10, 20 for image-caption similarity
* Formula for [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
* Bounding it by k just means that we will only score a caption if it appears in the most likely captions for the image.

| Experiment                          | k=1     | k=3     | k=5     | k=10    | k=20    |
|-------------------------------------|---------|---------|---------|---------|---------|
| baseline                            | 0.42580 | 0.53402 | 0.55837 | 0.57349 | 0.57829 |
| [run-1](src/train_configs/run1.cfg) | 0.69130 | 0.78962 | 0.80113 | 0.80517 | 0.80589 |
| [run-2](src/train_configs/run2.cfg) | 0.71200 | 0.80445 | 0.81519 | 0.81912 | 0.81968 |
| [run-3](src/train_configs/run3.cfg) | 0.34540 | 0.46338 | 0.49253 | 0.51154 | 0.51753 |
| [run-4](src/train_configs/run4.cfg) | 0.78760 | 0.86227 | 0.86870 | 0.87080 | 0.87120 |
| [run-5](src/train_configs/run5.cfg) | **0.80200** | **0.87170** | **0.87743** | **0.87966** | **0.88002** |


## Image Search

The demo is on a standalone CPU-only box since we are only doing inference. The corpus of images + captions used is the combination of the training, validation, and unseen test sets provided by ImageCLEF 2017 Caption Prediction challenge. The captions and image vectors are hosted on the Vespa search engine, which provides both BM25 based text search services and HNSW and Cosine similarity based Approximate Nearest Neighbor services.

### Environment

* AWS EC2 r5.2xlarge
  * 8 CPUs
  * 64 GB RAM
  * 200 GB disk
* Ubuntu 18.04 (bionic)
* Anaconda 2021.05-Linux
  * Python 3.7.10
  * PyTorch 1.9.0
  * SpaCy (with `en_core_web_sm` Language Model)
  * Transformers
  * Streamlit

### Vespa

  * Docker -- [instructions](https://docs.docker.com/engine/install/ubuntu/)
  * Vespa -- [instructions](https://docs.vespa.ai/en/vespa*quick*start.html)
    * download [vespa-engine/sample-apps](https://github.com/vespa-engine/sample-apps)
    * create new app `clip-demo` as new app in `sample-apps`
    * copy `vespa/src` folder to `sample-apps/clip-demo`
    * download [sujitpal/vespa-poc](https://github.com/sujitpal/vespa-poc)
    * update scripts in `bash-scripts` to point to `clip-demo` sample app
    * `launch.sh` to start docker instance
    * `deploy.sh` to deploy `clip-demo` to Vespa
    * `status.sh` to verify Vespa status
  * Prepare data
    * Images -- `mkdir CaptionPrediction; aws s3 cp --recursive s3://els-corpora/ImageCLEF/ImageCLEF2017/Caption/CaptionPrediction .`
      * unzip image collectiosn under `training`, `validation`, `test`
    * Vectors -- `cd CaptionPrediction; aws s3 cp --recursive s3://els-ats/sujit/clip-imageclef/vectors .`
    * Models -- `mkdir clip-model; cd clip-model; aws s3 cp --recursive s3://els-ats/sujit/clip-imageclef/models .`
  * Load data -- run [01-load-index.py](vespa/01-load-index.py)

### Streamlit

* A Streamlit based demo illustrates the usage of the trained model for the following use cases. 

  * text to image search -- user enters a text query to match against image corpus
    * using caption text -- this is standard text query, searches images by caption text
    * using query text vector -- since CLIP learns to embed images close to their corresponding captions, we do a vector search against the image corpus using the vector representation of the query text. Distance measure uses Cosine Similarity and Vespa provides HNSW based Approximate Nearest Neighbor search.
    * using hybrid text + vector search -- relevance is a linear interpolation of the BM25 relevance from caption search and cosine similarity from vector search.
  * image to image search -- this is more like a search for similar images in the corpus, user provides an image ID and search will return similar images to the query image in the corpus. We could also query the corpus using an external image with our infrastructure (compute image embedding from trained model and find nearest neighbors in the corpus), but the demo does not support that functionality.
    * image only -- use consine similarity between the query image vector and vectors for images in the corpus
    * image + text -- use hybrid text + vector search, computing relevance as a linear interpolation of cosine similarity between image vectors and BM25 similarity between source image caption and target image captions.

* To run streamlit server: 
  * `streamlit run app.py` on server
  * open tunnel for port 8501 to localhost: `ssh CLIPDemo -L 8501:localhost:8501`
  * bring up demo on local browser: `http://localhost:8501`

### Automatic Startup and Shutdown

Ideally, one should be able to switch the machine on in EC2, and the services should just start up by themselves, and vice-versa when the machine is swtiched off. While the machine is switched on, a user should be able to access the Demo at [http://10.169.23.142/clip-demo](http://10.169.23.142/clip-demo) as long as they are inside the Elsevier VPN.

This requires the following services to be turned on and off automatically.

#### Vespa

To install, run the following commands (note: this depends on the [vespa-poc](https://github.com/sujitpal/vespa-poc), specifically the `bash-scripts` folder.

* `cd $HOME`
* `git clone https://github.com/sujitpal/vespa-poc.git`
* `cd $PROJECT_ROOT/vespa`
* `sudo cp vespa.service /etc/systemd/system/`
* `sudo systemctl start vespa`
* `sudo systemctl enable vespa`

#### Streamlit

To install, run the following commands.

* `cd $PROJECT_ROOT/demo`
* `sudo cp streamlit.service /etc/systemd/system/`
* `sudo systemctl start streamlit`
* `sudo systemctl enable streamlit`

#### Ngnix

We need to install Nginx and enable reverse-proxy for streamlit so it is generally accessible within the Elsevier AWS VPN.

* Needs `HTTP over DC` security group to be enabled.
* `cd $PROJECT_ROOT/demo`
* Copy the snippet inside `streamlit.nginx` inside the server block of `/etc/nginx/sites-enabled/default`
* `sudo nginx -t`
* `sudo systemctl start nginx`
* Streamlit app should be accessible at `http://${PRIVATE_IP_ADDRESS}/clip-demo`



