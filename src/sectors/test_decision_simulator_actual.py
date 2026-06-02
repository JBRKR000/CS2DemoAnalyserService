from sectors.decision_simulator import simulate_decisions


def test_mixed_killed_fallback():
    entry = {
        "round_num": 16,
        "side": "T",
        "review_type": "mixed",
        "summary": "killed jabbi with m4a1 in a round-swinging duel.",
        "reasons": ["-21.98 pp ML impact"],
        "ml_impact": -0.2198,
    }
    results = simulate_decisions({}, [entry])
    assert results, "simulate_decisions returned no results"
    actual = results[0].get("actual_summary", "")
    assert not actual.lower().startswith("killed"), f"Actual wrongly starts with 'killed': {actual}"


if __name__ == '__main__':
    test_mixed_killed_fallback()
    print('test passed')
