import numpy as np
import cv2

import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from pandas.core.common import flatten

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

from visualizations import *

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CHANNELS = 3
EMBEDDING_SIZE = 512


def prepare_data():
    df = pd.read_csv('./dataset/styles.csv', on_bad_lines='skip')
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    df = df.reset_index(drop=True)

    return df


def get_image_path(file_name):
    return f'./dataset/images/{file_name}'


def load_image(file_name):
    return cv2.imread(get_image_path(file_name))


def summarize_model(model):
    from torchsummary import summary
    summary(resnet, (CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))


def get_embedding(model, file_name):
    try:
        img = Image.open(get_image_path(file_name)).convert('RGB') 
    except FileNotFoundError:
        print(f'image not found: {file_name}')
        raise

    scale = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    standardize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # default values used on imagenet
    to_tensor = transforms.ToTensor()

    transformed_img = torch.Tensor(
        standardize(to_tensor(scale(img)).unsqueeze(0))
    )

    embedding = torch.zeros(EMBEDDING_SIZE)

    def copy_layer(_, __, output):
        embedding.copy_(output.data.reshape(output.data.size(1)))

    pooling_layer = model._modules.get('avgpool')
    attached_layer = pooling_layer.register_forward_hook(copy_layer)
    model(transformed_img)

    attached_layer.remove()

    return embedding


if __name__ == '__main__':
    df = prepare_data()

    image_name = df.iloc[0].image
    image = load_image(image_name)

    figures = {'im'+str(i): load_image(row.image) for i, row in df.sample(6).iterrows()}
    show_images(figures, 2, 3, filename='output/testimages.png')

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    summarize_model(resnet) 


    sample_embedding = get_embedding(resnet, df.iloc[0].image)
    print(sample_embedding)

    


