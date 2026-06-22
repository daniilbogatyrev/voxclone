from voxclone.eval.picker import combined_score, pick_best

WEIGHTS = {"similarity": 0.5, "naturalness": 0.3, "wer": 0.2}

def test_combined_score_rewards_sim_and_naturalness_penalizes_wer():
    # naturalness normalized by /5; wer enters as (1 - wer)
    s = combined_score(sim=1.0, utmos=5.0, wer=0.0, weights=WEIGHTS)
    assert abs(s - (0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0)) < 1e-6

def test_pick_best_chooses_highest_combined():
    metrics = {
        "ckpt_5":  {"similarity": 0.8, "naturalness": 4.0, "wer": 0.10},
        "ckpt_10": {"similarity": 0.9, "naturalness": 4.2, "wer": 0.05},
        "ckpt_15": {"similarity": 0.85, "naturalness": 3.0, "wer": 0.20},
    }
    assert pick_best(metrics, WEIGHTS) == "ckpt_10"

def test_combined_score_normalizes_sim_by_ceiling():
    full = combined_score(sim=0.6, utmos=5.0, wer=0.0, weights=WEIGHTS, ceiling=0.6)
    assert abs(full - (0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0)) < 1e-6
    half = combined_score(sim=0.6, utmos=5.0, wer=0.0, weights=WEIGHTS, ceiling=1.2)
    assert abs(half - (0.5 * 0.5 + 0.3 * 1.0 + 0.2 * 1.0)) < 1e-6

def test_pick_best_disqualifies_high_wer():
    metrics = {
        "loud_impostor": {"similarity": 0.99, "naturalness": 4.8, "wer": 0.50},
        "honest":        {"similarity": 0.70, "naturalness": 4.0, "wer": 0.05},
    }
    assert pick_best(metrics, WEIGHTS, wer_dq_threshold=0.20) == "honest"

def test_pick_best_all_disqualified_falls_back():
    metrics = {
        "a": {"similarity": 0.8, "naturalness": 4.0, "wer": 0.40},
        "b": {"similarity": 0.6, "naturalness": 3.0, "wer": 0.30},
    }
    assert pick_best(metrics, WEIGHTS, wer_dq_threshold=0.20) == "a"
