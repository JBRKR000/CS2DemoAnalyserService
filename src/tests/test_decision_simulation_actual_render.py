from sectors.decision_simulator import simulate_decisions


def test_killed_negative_replaced():
    report_data = {}
    vod_review_priority = [
        {
            "round_num": 1,
            "review_type": "mixed",
            "summary": "killed jabbi with m4a1 in a round-swinging duel.",
            "ml_impact": -0.52,
            "side": "CT",
        }
    ]

    results = simulate_decisions(report_data, vod_review_priority)
    assert results, "simulate_decisions returned no results"
    actual_summary = results[0]["actual_summary"]
    assert not actual_summary.strip().lower().startswith("killed"), (
        f"actual_summary should not start with 'killed', got: {actual_summary!r}"
    )


if __name__ == "__main__":
    test_killed_negative_replaced()
    print("test passed")
