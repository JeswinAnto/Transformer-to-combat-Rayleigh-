import numpy as np
import torch
from scipy.signal import stft
from scipy.interpolate import interp1d

def rayleigh_fading_with_doppler(num_symbols, f_d, fs):
    """Generate Rayleigh fading channel with Doppler effect."""
    t = np.arange(num_symbols) / fs
    w = (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) / np.sqrt(2)
    alpha = np.exp(-2 * np.pi * f_d / fs)
    h = np.zeros(num_symbols, dtype=np.complex128)
    h[0] = w[0]
    for n in range(1, num_symbols):
        h[n] = alpha * h[n-1] + np.sqrt(1 - alpha**2) * w[n]
    return h / np.sqrt(np.mean(np.abs(h)**2))

def qpsk_modulate(bits, fs, fc):
    """Modulate bits to QPSK symbols on a carrier."""
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    bits = bits.reshape(-1, 2)
    indices = 2 * bits[:, 0] + bits[:, 1]
    baseband = constellation[indices]
    t = np.arange(len(baseband)) / fs
    real_part = np.real(baseband) * np.cos(2 * np.pi * fc * t)
    imag_part = np.imag(baseband) * (-np.sin(2 * np.pi * fc * t))
    return real_part + imag_part, baseband

def qpsk_demodulate(symbols):
    """Demodulate QPSK symbols to bits."""
    decisions = np.zeros((len(symbols), 2), dtype=np.int32)
    I = np.real(symbols)
    Q = np.imag(symbols)
    decisions[:, 0] = (I < 0).astype(int)
    decisions[:, 1] = (Q < 0).astype(int)
    return decisions.flatten()

def qpsk_to_labels(symbols):
    """Map QPSK symbols to integer labels (0-3)."""
    labels = np.zeros(len(symbols), dtype=np.int64)
    I = np.real(symbols)
    Q = np.imag(symbols)
    labels[(I > 0) & (Q > 0)] = 0
    labels[(I > 0) & (Q < 0)] = 1
    labels[(I < 0) & (Q > 0)] = 2
    labels[(I < 0) & (Q < 0)] = 3
    return labels

def awgn_noise(symbols, snr_db):
    """Add AWGN noise to symbols based on SNR in dB."""
    snr = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(symbols)**2)
    noise_power = signal_power / snr
    if np.iscomplexobj(symbols):
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(len(symbols))
    return noise

def compute_per_symbol_features(received, transmitted, h, seq_len, pilot_pos=0, local_energy_window=5, 
                               stft_window_size=64, stft_hop_length=16, stft_num_freq_bins=32,
                               num_mag_bins=10, num_phase_bins=8, time_encoding_dim=10, fs=1000):
    """Compute features for transformer equalizer."""
    N = len(received)
    I = np.real(received)
    Q = np.imag(received)
    amp = np.abs(received)
    conj_prev = np.concatenate(([1.0+0j], np.conj(received[:-1])))
    dphase = np.angle(received * conj_prev)
    dphase[0] = dphase[1] if N > 1 else 0.0

    half = local_energy_window // 2
    padded_mag2 = np.pad(np.abs(received)**2, (half, half), mode='reflect')
    local_energy = np.array([padded_mag2[i:i+local_energy_window].mean() for i in range(N)])

    h_pilot_real = np.zeros(N)
    h_pilot_imag = np.zeros(N)
    for start in range(0, N, seq_len):
        pilot_idx = start + pilot_pos
        if pilot_idx >= N:
            break
        s_pilot = transmitted[pilot_idx]
        r_pilot = received[pilot_idx]
        h_est = (r_pilot / s_pilot) if np.abs(s_pilot) > 1e-12 else 1+0j
        end = min(start+seq_len, N)
        h_pilot_real[start:end] = np.real(h_est)
        h_pilot_imag[start:end] = np.imag(h_est)

    received_for_stft = np.real(received) if np.iscomplexobj(received) else received
    f, t, Zxx = stft(received_for_stft, fs=fs, nperseg=stft_window_size, 
                     noverlap=stft_window_size - stft_hop_length, 
                     nfft=stft_window_size, window='hann')
    stft_mags = np.abs(Zxx)[:stft_num_freq_bins, :]
    stft_mags_aligned = np.zeros((N, stft_num_freq_bins))
    symbol_times = np.arange(N) / fs
    for i in range(stft_num_freq_bins):
        if len(t) > 1:
            interpolator = interp1d(t, stft_mags[i], kind='linear', fill_value='extrapolate')
            stft_mags_aligned[:, i] = interpolator(symbol_times)
        else:
            stft_mags_aligned[:, i] = stft_mags[i, 0]

    h_mag = np.abs(h)
    h_phase = np.angle(h)
    mag_bins = np.quantile(h_mag, np.linspace(0, 1, num_mag_bins + 1))
    mag_indices = np.digitize(h_mag, mag_bins, right=True).clip(1, num_mag_bins) - 1
    phase_bins = np.linspace(0, 2 * np.pi, num_phase_bins + 1)
    phase_indices = np.digitize(h_phase, phase_bins, right=True).clip(1, num_phase_bins) - 1
    bin_indices = mag_indices * num_phase_bins + phase_indices
    bin_embedding = np.zeros((N, num_mag_bins * num_phase_bins))
    bin_embedding[np.arange(N), bin_indices] = 1
    mag_feature = (mag_bins[mag_indices] + mag_bins[mag_indices + 1]) / 2

    time_encoding = np.zeros((N, time_encoding_dim))
    div_term = np.exp(np.arange(0, time_encoding_dim, 2) * -(np.log(10000.0) / time_encoding_dim))
    time_encoding[:, 0::2] = np.sin(symbol_times[:, None] * div_term)
    time_encoding[:, 1::2] = np.cos(symbol_times[:, None] * div_term)

    feats = np.stack(
        [I, Q, amp, dphase, h_pilot_real, h_pilot_imag, local_energy, mag_feature] +
        [stft_mags_aligned[:, i] for i in range(stft_num_freq_bins)] +
        [bin_embedding[:, i] for i in range(num_mag_bins * num_phase_bins)] +
        [time_encoding[:, i] for i in range(time_encoding_dim)],
        axis=1
    )
    feats_tensor = torch.FloatTensor(feats)
    mean = feats_tensor.mean(dim=0, keepdim=True)
    std = feats_tensor.std(dim=0, keepdim=True) + 1e-9
    return (feats_tensor - mean).numpy() / std.numpy()