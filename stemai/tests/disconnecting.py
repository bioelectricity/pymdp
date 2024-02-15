import numpy as np


def test_disconnecting_B(original_B, new_B):
    assert (
        new_B[0].shape[0] == original_B[0].shape[0] // 2
    ), "The number of states should have decreased"
    assert (
        new_B[0].shape[1] == original_B[0].shape[1] // 2
    ), "The number of states should have decreased"
    assert (
        new_B[0].shape[2] == original_B[0].shape[2] // 2
    ), "The number of states should have decreased"

    print(f"Disconnecting B shape test passed ")
    # add some smart way to test that the marginal
