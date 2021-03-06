import numpy as np
"""Script for stochastic gradient descent
    Adapted from UMAP official implementation
"""
def optimize_embedding(head_embedding, tail_embedding, a, b, dim, n_epochs, graph, graph_rows, graph_cols):
    """
    Optimize embedding

    Parameters 
    -----------
    head_embedding : array
    tail_embedding : array
    a : float
    b : float
    dim : int
    n_epochs: int
    graph : matrix
    graph_rows : array
    graph_cols : array

    Returns
    --------
    head_embedding : array(n_samples, dimensions)
    """    
    alpha = 1.0
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
    epochs_per_negative_sample = epochs_per_sample / 5.0
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    head = graph_rows
    tail = graph_cols

    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
                if epoch_of_next_sample[i] <= n:
                    j = head[i]
                    k = tail[i]
                    
                    current = head_embedding[j]
                    other = tail_embedding[k]

                    dist_output, grad_dist_output = euclidean_grad(current, other)

                    if dist_output > 0.0:
                        w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                    else:
                        w_l = 1.0
                    grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

                    for d in range(dim):
                        grad_d = clip(grad_coeff * grad_dist_output[d])

                        current[d] += grad_d * alpha

                    epoch_of_next_sample[i] += epochs_per_sample[i]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
                    )

                    for p in range(n_neg_samples):
                        k = np.random.randint(1000) % graph.shape[1]

                        other = tail_embedding[k]

                        dist_output, grad_dist_output = euclidean_grad(
                            current, other
                        )

                        if dist_output > 0.0:
                            w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                        elif j == k:
                            continue
                        else:
                            w_l = 1.0

                        grad_coeff = 2 * b * w_l / (dist_output + 1e-6)

                        for d in range(dim):
                            grad_d = clip(grad_coeff * grad_dist_output[d])
                            current[d] += grad_d * alpha

                    epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i])
            
        alpha = alpha * (1.0 - (float(n) / float(n_epochs)))
    return head_embedding

def euclidean_grad(x, y):
        """Standard euclidean distance and its gradient."""
        result = 0.0
        for i in range(x.shape[0]):
            result += (x[i] - y[i]) ** 2
        d = np.sqrt(result)
        grad = (x - y) / (1e-6 + d)
        return d, grad

def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.
    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.
    n_epochs: int
        The total number of epochs we want to train for.
    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result

def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)
    Parameters
    ----------
    val: float
       
    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val
