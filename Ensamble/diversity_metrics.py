import numpy as np
# Calculate pairwise disagreement between models
def disagreement_rate(pred_list):
    n_models = pred_list.shape[1]
    disagreement_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagreement = np.mean(pred_list[:, i] != pred_list[:, j])
            disagreement_matrix[i, j] = disagreement
            disagreement_matrix[j, i] = disagreement  # Symmetric matrix

    return disagreement_matrix