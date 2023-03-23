import joblib

import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import pandas as pd
from pandas.core.common import flatten

from extract_features import *
from common import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def recommend(filename, model, embeddings, n=10):
    embedding = get_embedding(model, filename)
    print(embedding)
    similarity_scores = cosine_similarity(embedding.unsqueeze(0), embeddings)
    similarity_scores = list(flatten(similarity_scores))
    similarity_scores_df = pd.DataFrame(similarity_scores, columns=['Score'])
    similarity_scores_df = similarity_scores_df.sort_values(by=['Score'], ascending=False)

    print(similarity_scores_df['Score'][:10])

    topN = similarity_scores_df[:n].index
    topN = list(flatten(topN))
    images = list(flatten([df[df.index==i]['image'] for i in topN]))
    
    return images


if __name__ == '__main__':
    df = prepare_data()
    embeddings = joblib.load('output/embeddings.pkl')
    model = get_model(device)

    recommendations = recommend('output/56913.jpg', model, embeddings)
    show_recommendations('output/56913.jpg', recommendations, 'output/recommendations.png')
