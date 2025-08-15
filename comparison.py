import numpy as np
import torch
import torch.nn as nn
from transformer import TransformerEqualizer
import matplotlib.pyplot as plt
from utils import rayleigh_fading_with_doppler, qpsk_modulate, qpsk_demodulate, qpsk_to_labels, awgn_noise, compute_per_symbol_features
import os
from datetime import datetime

# Create plots directory
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)
current_date = datetime.now().strftime("%Y-%m-%d")

# Simulation parameters
CONFIG = {
    'seq_len': 3,
    'num_symbols': 10239,
    'snr_db': 20,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'batch_size': 16,
    'fs': 10000,
    'fc': 2000,
    'fd_list': [5, 50, 100],
    'stft_window_size': 32,
    'stft_hop_length': 8,
    'stft_num_freq_bins': 17,
    'num_mag_bins': 10,
    'num_phase_bins': 8,
    'time_encoding_dim': 20,
    'tap_length': 5  # For LMMSE, DFE, RLS
}

# Equalizer implementations
class LMMSEEqualizer:
    """Linear Minimum Mean Square Error (LMMSE) Equalizer."""
    def __init__(self, tap_length):
        self.tap_length = tap_length
    
    def train(self, received, transmitted, snr_db):
        N = len(received)
        R = np.zeros((self.tap_length, self.tap_length), dtype=np.complex128)
        p = np.zeros(self.tap_length, dtype=np.complex128)
        noise_power = np.mean(np.abs(transmitted)**2) / (10 ** (snr_db / 10))
        
        for i in range(self.tap_length, N):
            x = received[i-self.tap_length:i][::-1]
            R += np.outer(x, np.conj(x))
            p += transmitted[i] * np.conj(x)
        R /= (N - self.tap_length)
        p /= (N - self.tap_length)
        R += noise_power * np.eye(self.tap_length)  # Regularize with noise power
        self.weights = np.linalg.solve(R, p)
    
    def equalize(self, received):
        N = len(received)
        equalized = np.zeros(N, dtype=np.complex128)
        for i in range(self.tap_length, N):
            x = received[i-self.tap_length:i][::-1]
            equalized[i] = np.dot(self.weights, x)
        return equalized[self.tap_length:]

class DFE:
    """Decision Feedback Equalizer."""
    def __init__(self, ff_taps, fb_taps, learning_rate=0.01):
        self.ff_taps = ff_taps
        self.fb_taps = fb_taps
        self.ff_weights = np.zeros(ff_taps, dtype=np.complex128)
        self.fb_weights = np.zeros(fb_taps, dtype=np.complex128)
        self.lr = learning_rate
    
    def train(self, received, transmitted):
        N = len(received)
        decisions = np.zeros(N, dtype=np.complex128)
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        
        for i in range(max(self.ff_taps, self.fb_taps), N):
            ff_input = received[i-self.ff_taps:i][::-1]
            fb_input = decisions[i-self.fb_taps:i][::-1]
            output = np.dot(self.ff_weights, ff_input) + np.dot(self.fb_weights, fb_input)
            decision = constellation[np.argmin(np.abs(constellation - output))]
            error = transmitted[i] - output
            self.ff_weights += self.lr * error * np.conj(ff_input)
            self.fb_weights += self.lr * error * np.conj(fb_input)
            decisions[i] = decision
    
    def equalize(self, received):
        N = len(received)
        decisions = np.zeros(N, dtype=np.complex128)
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        
        for i in range(max(self.ff_taps, self.fb_taps), N):
            ff_input = received[i-self.ff_taps:i][::-1]
            fb_input = decisions[i-self.fb_taps:i][::-1]
            output = np.dot(self.ff_weights, ff_input) + np.dot(self.fb_weights, fb_input)
            decisions[i] = constellation[np.argmin(np.abs(constellation - output))]
        return decisions[max(self.ff_taps, self.fb_taps):]

class RLSEqualizer:
    """Recursive Least Squares (RLS) Equalizer."""
    def __init__(self, tap_length, forgetting_factor=0.99):
        self.tap_length = tap_length
        self.forgetting_factor = forgetting_factor
        self.P = np.eye(tap_length) / 0.01
        self.weights = np.zeros(tap_length, dtype=np.complex128)
    
    def train(self, received, transmitted):
        N = len(received)
        for i in range(self.tap_length, N):
            x = received[i-self.tap_length:i][::-1]
            k = np.dot(self.P, x) / (self.forgetting_factor + np.dot(x.conj(), np.dot(self.P, x)))
            error = transmitted[i] - np.dot(self.weights, x)
            self.weights += k * error.conj()
            self.P = (self.P - np.outer(k, np.dot(x.conj(), self.P))) / self.forgetting_factor
    
    def equalize(self, received):
        N = len(received)
        equalized = np.zeros(N, dtype=np.complex128)
        for i in range(self.tap_length, N):
            x = received[i-self.tap_length:i][::-1]
            equalized[i] = np.dot(self.weights, x)
        return equalized[self.tap_length:]

def prepare_data(fd):
    """Generate and preprocess QPSK data for a given Doppler frequency."""
    bits_all = np.random.randint(0, 2, CONFIG['num_symbols'] * 2)
    _, transmitted_baseband = qpsk_modulate(bits_all, CONFIG['fs'], CONFIG['fc'])
    h = rayleigh_fading_with_doppler(CONFIG['num_symbols'], fd, CONFIG['fs'])
    received_baseband = transmitted_baseband * h + awgn_noise(transmitted_baseband, CONFIG['snr_db'])
    
    feats = compute_per_symbol_features(
        received_baseband, transmitted_baseband, h, CONFIG['seq_len'],
        pilot_pos=0, local_energy_window=5, stft_window_size=CONFIG['stft_window_size'],
        stft_hop_length=CONFIG['stft_hop_length'], stft_num_freq_bins=CONFIG['stft_num_freq_bins'],
        num_mag_bins=CONFIG['num_mag_bins'], num_phase_bins=CONFIG['num_phase_bins'],
        time_encoding_dim=CONFIG['time_encoding_dim'], fs=CONFIG['fs']
    )
    
    num_blocks = len(feats) // CONFIG['seq_len']
    feats = feats[:num_blocks * CONFIG['seq_len']].reshape(num_blocks, CONFIG['seq_len'], -1)
    transmitted_baseband = transmitted_baseband[:num_blocks * CONFIG['seq_len']]
    bits_all = bits_all[:num_blocks * CONFIG['seq_len'] * 2]
    transmitted_labels = qpsk_to_labels(transmitted_baseband).reshape(num_blocks, CONFIG['seq_len'])
    bits_per_seq = bits_all.reshape(num_blocks, CONFIG['seq_len'] * 2)
    
    received_tensor = torch.FloatTensor(feats)
    labels_tensor = torch.LongTensor(transmitted_labels)
    
    num_sequences = len(received_tensor)
    train_size = int(0.7 * num_sequences)
    val_size = int(0.15 * num_sequences)
    
    train_data = (received_tensor[:train_size], labels_tensor[:train_size], bits_per_seq[:train_size])
    test_data = {
        'received': received_tensor[train_size + val_size:],
        'labels': labels_tensor[train_size + val_size:],
        'bits': bits_per_seq[train_size + val_size:],
        'transmitted': transmitted_baseband[(train_size + val_size) * CONFIG['seq_len']:],
        'received_baseband': received_baseband[(train_size + val_size) * CONFIG['seq_len']:]
    }
    
    return train_data, test_data, feats.shape[-1], received_baseband[:num_blocks * CONFIG['seq_len']], transmitted_baseband

def train_transformer(train_data, feat_dim):
    """Train the transformer equalizer."""
    train_received, train_labels, _ = train_data
    transformer = TransformerEqualizer(feat_dim=feat_dim, d_model=16, nhead=1, num_layers=1, dim_feedforward=64)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(CONFIG['num_epochs']):
        transformer.train()
        total_loss = 0.0
        num_batches = 0
        for i in range(0, len(train_received), CONFIG['batch_size']):
            batch_received = train_received[i:i + CONFIG['batch_size']]
            batch_labels = train_labels[i:i + CONFIG['batch_size']]
            optimizer.zero_grad()
            output = transformer(batch_received)
            loss = criterion(output.view(-1, 4), batch_labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    
    return transformer

def evaluate_equalizers(fd_list, equalizers, data_dict):
    """Evaluate all equalizers and compute BER/SER."""
    metrics = {name: {fd: {'ber': None, 'ser': None} for fd in fd_list} for name in equalizers.keys()}
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    for fd in fd_list:
        test_data, received_baseband, transmitted_baseband = data_dict[fd]['test_data'], data_dict[fd]['received_baseband'], data_dict[fd]['transmitted']
        
        for name, equalizer in equalizers.items():
            if name == 'Transformer':
                transformer = equalizer
                transformer.eval()
                with torch.no_grad():
                    output = transformer(test_data['received'])
                    predicted_classes = torch.argmax(output, dim=-1).view(-1).numpy()
                    equalized = constellation[predicted_classes]
                    test_labels = test_data['labels'].view(-1).numpy()
                    metrics[name][fd]['ser'] = np.mean(predicted_classes != test_labels)
                    metrics[name][fd]['ber'] = np.mean(qpsk_demodulate(equalized) != test_data['bits'].reshape(-1))
            else:
                equalized = equalizer.equalize(received_baseband)
                equalized = equalized[:len(test_data['transmitted'])]
                predicted_classes = np.argmin(np.abs(constellation[:, None] - equalized[None, :]), axis=0)
                test_labels = qpsk_to_labels(test_data['transmitted'])
                metrics[name][fd]['ser'] = np.mean(predicted_classes != test_labels)
                metrics[name][fd]['ber'] = np.mean(qpsk_demodulate(equalized) != test_data['bits'].reshape(-1))
    
    return metrics

def plot_comparison(metrics, fd_list):
    """Plot BER and SER comparison across equalizers."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name in metrics.keys():
        plt.plot(fd_list, [metrics[name][fd]['ber'] for fd in fd_list], marker='o', label=name)
    plt.title('BER vs Doppler Frequency')
    plt.xlabel('Doppler Frequency (Hz)')
    plt.ylabel('BER')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name in metrics.keys():
        plt.plot(fd_list, [metrics[name][fd]['ser'] for fd in fd_list], marker='o', label=name)
    plt.title('SER vs Doppler Frequency')
    plt.xlabel('Doppler Frequency (Hz)')
    plt.ylabel('SER')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/equalizer_comparison_{current_date}.png")
    plt.close()

# Main execution
data_dict = {}
equalizers = {
    'LMMSE': LMMSEEqualizer(CONFIG['tap_length']),
    'DFE': DFE(ff_taps=CONFIG['tap_length'], fb_taps=CONFIG['tap_length']),
    'RLS': RLSEqualizer(CONFIG['tap_length']),
    'Transformer': None
}

for fd in CONFIG['fd_list']:
    train_data, test_data, feat_dim, received_baseband, transmitted_baseband = prepare_data(fd)
    data_dict[fd] = {
        'test_data': test_data,
        'received_baseband': received_baseband,
        'transmitted': transmitted_baseband
    }
    
    # Train equalizers
    equalizers['LMMSE'].train(received_baseband[:len(train_data[0]) * CONFIG['seq_len']], transmitted_baseband[:len(train_data[0]) * CONFIG['seq_len']], CONFIG['snr_db'])
    equalizers['DFE'].train(received_baseband[:len(train_data[0]) * CONFIG['seq_len']], transmitted_baseband[:len(train_data[0]) * CONFIG['seq_len']])
    equalizers['RLS'].train(received_baseband[:len(train_data[0]) * CONFIG['seq_len']], transmitted_baseband[:len(train_data[0]) * CONFIG['seq_len']])
    equalizers['Transformer'] = train_transformer(train_data, feat_dim)

# Evaluate and plot
metrics = evaluate_equalizers(CONFIG['fd_list'], equalizers, data_dict)
plot_comparison(metrics, CONFIG['fd_list'])

# Print summary
print("\n=== Equalizer Comparison ===")
print("Equalizer | Doppler (Hz) | BER | SER")
print("-" * 40)
for name in equalizers.keys():
    for fd in CONFIG['fd_list']:
        print(f"{name:<9} | {fd:>12} | {metrics[name][fd]['ber']:.6f} | {metrics[name][fd]['ser']:.6f}")

print(f"\nComparison completed! SNR: {CONFIG['snr_db']} dB, Symbols: {CONFIG['num_symbols']}")