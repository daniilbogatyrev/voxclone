import random
from voxclone.common.seed import set_seed

def test_set_seed_makes_random_deterministic():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b
