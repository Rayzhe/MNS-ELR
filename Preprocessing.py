import numpy as np
import scipy.linalg

def process_eeg_covariance(dataTrain, dataTest):
    """
    Process EEG data: compute normalized covariance, whitening, and matrix logarithm mapping.
    Args:
        dataTrain: (n_trials, n_samples, n_channels)
        dataTest:  (n_trials, n_samples, n_channels)
    Returns:
        RsTrain_log: (n_trials, n_channels, n_channels)
        RsTest_log:  (n_trials, n_channels, n_channels)
    """

    def get_covariance(data):
        """Compute normalized covariance matrix for each trial."""
        num_trials, num_samples, num_channels = data.shape
        cov = np.zeros((num_trials, num_channels, num_channels))
        for i in range(num_trials):
            signal_epoch = data[i]  # (n_samples, n_channels)
            cov_i = signal_epoch.T @ signal_epoch  # (n_channels, n_channels)
            cov[i] = cov_i / np.trace(cov_i)
        return cov

    # 1. Compute normalized covariance matrices
    RsTrain = get_covariance(dataTrain)
    RsTest = get_covariance(dataTest)

    # 2. Compute the mean covariance matrix (training set)
    n_trials, n_samples, n_channels = dataTrain.shape
    cov_matrices = np.zeros((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        sample = dataTrain[i, :, :].T  # (n_channels, n_samples)
        cov_matrices[i] = np.cov(sample)
    average_cov_matrix = np.mean(cov_matrices, axis=0)

    # 3. Calculate the whitening matrix (inverse square root of the mean covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(average_cov_matrix)
    Rp_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    # 4. Whiten the covariance matrices
    for i in range(RsTrain.shape[0]):
        RsTrain[i] = Rp_inv_sqrt @ RsTrain[i] @ Rp_inv_sqrt
    for i in range(RsTest.shape[0]):
        RsTest[i] = Rp_inv_sqrt @ RsTest[i] @ Rp_inv_sqrt

    # 5. Matrix logarithm mapping for each whitened covariance matrix
    for i in range(len(RsTrain)):
        RsTrain[i, :, :] = scipy.linalg.logm(RsTrain[i, :, :], disp=True)
    for i in range(len(RsTest)):
        RsTest[i, :, :] = scipy.linalg.logm(RsTest[i, :, :], disp=True)

    return RsTrain, RsTest

# Example usage:
# RsTrain_log, RsTest_log = process_eeg_covariance(dataTrain, dataTest)
