
import numpy as np
from pymdp import utils


def create_one_hot_A_matrix(A):
    """
    Convert the given A matrix into a one-hot encoded version.
    
    Parameters:
    A (numpy.ndarray): The input A matrix with shape (num_obs, state_factor_1, state_factor_2)
    
    Returns:
    numpy.ndarray: The one-hot encoded A matrix
    """

    new_A = utils.obj_array(len(A))
    for modality in range(len(A)):

    # Create a one-hot encoded version of A
        A_one_hot = np.zeros_like(A[modality])
        
        # Find the index of the maximum probability for each observation
        max_indices = np.argmax(A[modality], axis=0)
        
        # Set the corresponding element to 1
        it = np.nditer(max_indices, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            obs_idx = max_indices[idx]
            A_one_hot[(obs_idx,) + idx] = 1
            it.iternext()
        new_A[modality] = A_one_hot

    return new_A

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two distributions p and q.
    
    Parameters:
    p (numpy.ndarray): The first distribution.
    q (numpy.ndarray): The second distribution (typically the one-hot distribution).
    
    Returns:
    numpy.ndarray: The KL divergence for each element.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-10, 1)  # Avoid log(0)
    q = np.clip(q, 1e-10, 1)  # Avoid log(0)
    return np.sum(p * np.log(p / q), axis=0)


def measure_one_hotness(A):
    """
    Measure the one-hot-ness of the given A matrix for each modality and state factor.
    
    Parameters:
    A (list of numpy.ndarray): The list of A matrices for each modality.
    
    Returns:
    list of numpy.ndarray: The one-hot-ness scores for each modality and state factor.
    """
    one_hotness_scores = []
    one_hot_A = create_one_hot_A_matrix(A)

    
    for modality in range(len(A)):
        A_mod = A[modality]
        A_one_hot = one_hot_A[modality]
        kl_scores = kl_divergence(A_mod, A_one_hot)
        one_hotness_scores.append(kl_scores)
    
    return one_hotness_scores


def build_uniform_B_matrix(num_states, num_actions, noise_scale=0.05):
    num_factors = len(num_states)
    num_controls = len(num_actions)

    B = np.empty(num_factors, dtype=object)
    
    for factor in range(num_factors):
        B[factor] = np.zeros([num_states[factor], num_states[factor]] + num_actions)

        for c in range(num_controls):
            #slicing_tuple = [slice(None)] * B[factor].ndim

            for a in range(num_actions[c]):
                #slicing_tuple[num_factors + c + 1] = a

                # Create a fuzzy uniform distribution by adding noise to the uniform distribution
                noise = np.random.uniform(-noise_scale, noise_scale, size=B[factor][:,:,a].shape)
                uniform_vec = np.full(
                        (num_states[factor], num_states[factor]), 1 / num_states[factor]
                    )
                
                noisy_vec = uniform_vec + noise

                print(f"Noisy vec: {noisy_vec.shape}")
                B[factor][:, :, a] = noisy_vec / np.sum(noisy_vec,axis =0)
        
    assert utils.is_normalized(B), "B matrix is not normalized."
    
    return B


def build_uniform_A_matrix(num_obs, num_states):
    num_modalities = len(num_obs)
    num_factors = len(num_states)
    A = utils.obj_array(num_modalities)

    for modality in range(num_modalities):
        A[modality] = np.random.rand(*([num_obs[modality]] + num_states)) 
        A[modality] /= A[modality].sum(axis=0, keepdims=True)
    assert utils.is_normalized(A)
    return A