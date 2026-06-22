import numpy as np
from voxclone.eval.similarity import cosine, reference_ceiling, similarity_score

def test_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine(v, v) - 1.0) < 1e-6

def test_cosine_orthogonal_is_zero():
    assert abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-6

def test_reference_ceiling_is_mean_pairwise():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # pairs: (0,1)=1.0, (0,2)=0.0, (1,2)=0.0 -> mean = 1/3
    assert abs(reference_ceiling(emb) - 1 / 3) < 1e-6

def test_similarity_score_is_mean_to_real_centroid():
    gen = np.array([[1.0, 0.0], [1.0, 0.0]])
    real = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert abs(similarity_score(gen, real) - 1.0) < 1e-6
