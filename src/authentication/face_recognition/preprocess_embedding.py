import numpy as np
def preprocess_embedding(embedding):
    embedding = embedding - np.mean(embedding)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding