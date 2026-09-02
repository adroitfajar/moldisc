import numpy as np

from SMILESX.model import LSTMAttModel


def _weights(seed):
    model = LSTMAttModel.create(
        input_tokens=12,
        vocab_size=20,
        embed_units=8,
        lstm_units=4,
        tdense_units=8,
        dense_depth=2,
        random_seed=seed,
    )
    return model.get_weights()


def test_seeded_initialization_is_reproducible_and_run_specific():
    first = _weights(123)
    repeated = _weights(123)
    different_run = _weights(124)

    assert all(np.array_equal(left, right) for left, right in zip(first, repeated))
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(first, different_run)
    )

