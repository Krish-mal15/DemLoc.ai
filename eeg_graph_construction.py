import numpy as np
from scipy.signal import hilbert
from scipy.signal import welch

def compute_plv(eeg_data):
    n_channels, n_samples = eeg_data.shape

    # Compute the analytic signal using the Hilbert transform
    analytic_signal = np.apply_along_axis(lambda x: np.angle(hilbert(x)), axis=1, arr=eeg_data)

    # Initialize the PLV matrix
    plv_matrix = np.zeros((n_channels, n_channels))

    # Compute PLV between each pair of channels
    for i in range(n_channels):
        for j in range(n_channels):
            phase_diff = np.exp(1j * (analytic_signal[i] - analytic_signal[j]))
            plv_matrix[i, j] = np.abs(np.mean(phase_diff))

    return plv_matrix

def compute_psd(eeg_signal, sampling_frequency):
    psd_data = []
    for channel_data in eeg_signal:
        freqs, psd = welch(channel_data, fs=sampling_frequency, nperseg=512)
        psd_data.append(psd)
    psd_data = np.array(psd_data)
    psd_corr_matrix = np.corrcoef(psd_data)
    
    return psd_corr_matrix


def eeg_sim_matrix_calc(eeg_data, sfreq):
    plv_matrix = compute_plv(eeg_data)
    psd_corr_matrix = compute_psd(eeg_data, sampling_frequency=sfreq)

    # combined_matrix = (plv_matrix + psd_corr_matrix) / 2
    # combined_matrix = psd_corr_matrix
    combined_matrix = plv_matrix

    return combined_matrix