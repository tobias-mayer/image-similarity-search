import os.path

import numpy as np
import pandas as pd

import joblib
import swifter

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

from common import *

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CHANNELS = 3
EMBEDDING_SIZE = 512

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_embedding(model, filepath):
    try:
        img = Image.open(filepath).convert('RGB') 
    except:
        print(f'image not found: {filepath}')
        return torch.zeros(EMBEDDING_SIZE)

    scale = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    standardize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # default values used on imagenet
    to_tensor = transforms.ToTensor()

    transformed_img = standardize(to_tensor(scale(img))).unsqueeze(0)
    transformed_img = transformed_img.to(device)

    embedding = torch.zeros(EMBEDDING_SIZE)

    def copy_layer(model, input, output):
        embedding.copy_(output.data.reshape(EMBEDDING_SIZE))

    pooling_layer = model._modules.get('avgpool')
    attached_layer = pooling_layer.register_forward_hook(copy_layer)
    model(transformed_img)
    attached_layer.remove()

    return embedding


def get_similarity(embedding1, embedding2):
    return nn.CosineSimilarity()(embedding1, embedding2)


if __name__ == '__main__':
    df = prepare_data()

    image_name = df.iloc[0].image
    image = load_image(image_name)

    model = get_model(device)

    embedding1 = get_embedding(model, get_image_path(df.iloc[1].image)).reshape((1, -1))
    embedding2 = get_embedding(model, get_image_path(df.iloc[1000].image)).reshape((1, -1))

    figures = {'im'+str(i): load_image(row.image) for i, row in df.iloc[[1, 1000]].iterrows()}
    show_images(figures, 1, 2, filename='output/testimages.png')

    cos_sim = get_similarity(embedding1, embedding2)
    print(f'cosine similarity: {cos_sim}')


    # todo: run in batches
    embeddings = df['image'].swifter.apply(lambda img: get_embedding(model, get_image_path(img)))
    embeddings = embeddings.apply(pd.Series)

    joblib.dump(embeddings, 'output/embeddings.pkl', 9)

