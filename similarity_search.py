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
from torch.autograd import Variable

from PIL import Image

from visualizations import *


def prepare_data():
    df = pd.read_csv('./dataset/styles.csv', on_bad_lines='skip')
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    df = df.reset_index(drop=True)

    return df


def get_image_path(file_name):
    return f'./dataset/images/{file_name}'


def load_image(file_name):
    return cv2.imread(get_image_path(file_name))

df = prepare_data()

image_name = df.iloc[0].image
image = load_image(image_name)

figures = {'im'+str(i): load_image(row.image) for i, row in df.sample(6).iterrows()}
show_images(figures, 2, 3, filename='testimages.png')

