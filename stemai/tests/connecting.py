import numpy as np

def test_connecting_B(original_B, new_B):
    assert original_B[0].shape[0] == new_B[0].shape[0] // 2, "The number of states should have decreased"
    assert original_B[0].shape[1] == new_B[0].shape[1] // 2, "The number of states should have decreased"
    assert original_B[0].shape[2] == new_B[0].shape[2] // 2, "The number of states should have decreased"

    print(f"Connecting B shape test passed ")
    #add some smart way to test that the marginal