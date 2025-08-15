import numpy as np
import torch
import torch.nn as nn
from transformer import TransformerEqualizer
import os
from datetime import datetime
from utils import rayleigh_fading_with_doppler, qpsk_modulate, qpsk_to_labels, awgn_noise, compute_per_symbol_features, qpsk_demodulate
from ber_evaluation import evaluate_test_set

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
    'num_epochs': 100,
    'batch_size': 16,
    'fs': 10000,
    'fc': 2000,
    'fd_list': [5, 50, 100],
    'stft_window_size': 32,
    'stft_hop_length': 8,
    'stft_num_freq_bins': 17,
    'num_mag_bins': 10,
    'num_phase_bins': 8,
    'time_encoding_dim': 20
}

# Metrics storage
metrics = {
    fd: {'train_loss': [], 'val_loss': [], 'val_ber': [], 'val_ser': []}
    for fd in CONFIG['fd_list']
}
test_data = {}

def prepare_data(fd):
    """Generate and preprocess QPSK data for a given Doppler frequency."""
    print(f"\n=== Doppler Frequency: {fd} Hz ===")
    print("Generating QPSK data...")
    bits_all = np.random.randint(0, 2, CONFIG['num_symbols'] * 2)
    transmitted_real, transmitted_baseband = qpsk_modulate(bits_all, CONFIG['fs'], CONFIG['fc'])
    
    print("Applying Rayleigh fading channel...")
    h = rayleigh_fading_with_doppler(CONFIG['num_symbols'], fd, CONFIG['fs'])
    received_baseband = transmitted_baseband * h + awgn_noise(transmitted_baseband, CONFIG['snr_db'])
    
    print("Computing features...")
    feats = compute_per_symbol_features(
        received_baseband, transmitted_baseband, h, CONFIG['seq_len'],
        pilot_pos=0, local_energy_window=5, stft_window_size=CONFIG['stft_window_size'],
        stft_hop_length=CONFIG['stft_hop_length'], stft_num_freq_bins=CONFIG['stft_num_freq_bins'],
        num_mag_bins=CONFIG['num_mag_bins'], num_phase_bins=CONFIG['num_phase_bins'],
        time_encoding_dim=CONFIG['time_encoding_dim'], fs=CONFIG['fs']
    )
    
    # Align data into sequences
    num_blocks = len(feats) // CONFIG['seq_len']
    feats = feats[:num_blocks * CONFIG['seq_len']].reshape(num_blocks, CONFIG['seq_len'], -1)
    transmitted_baseband = transmitted_baseband[:num_blocks * CONFIG['seq_len']]
    bits_all = bits_all[:num_blocks * CONFIG['seq_len'] * 2]
    transmitted_labels = qpsk_to_labels(transmitted_baseband).reshape(num_blocks, CONFIG['seq_len'])
    bits_per_seq = bits_all.reshape(num_blocks, CONFIG['seq_len'] * 2)
    
    # Convert to tensors
    received_tensor = torch.FloatTensor(feats)
    labels_tensor = torch.LongTensor(transmitted_labels)
    
    # Split train/val/test
    num_sequences = len(received_tensor)
    train_size = int(0.7 * num_sequences)
    val_size = int(0.15 * num_sequences)
    
    train_data = (received_tensor[:train_size], labels_tensor[:train_size], bits_per_seq[:train_size])
    val_data = (
        received_tensor[train_size:train_size + val_size],
        labels_tensor[train_size:train_size + val_size],
        bits_per_seq[train_size:train_size + val_size]
    )
    test_data_fd = {
        'received': received_tensor[train_size + val_size:],
        'labels': labels_tensor[train_size + val_size:],
        'bits': bits_per_seq[train_size + val_size:],
        'transmitted': transmitted_baseband[(train_size + val_size) * CONFIG['seq_len']:]
    }
    
    print(f"Feature shape: {received_tensor.shape}")
    print(f"Label shape: {labels_tensor.shape}")
    return train_data, val_data, test_data_fd, feats.shape[-1]

def train_model(train_data, val_data, feat_dim):
    """Train the transformer equalizer model."""
    train_received, train_labels, train_bits = train_data
    val_received, val_labels, val_bits = val_data
    
    transformer = TransformerEqualizer(feat_dim=feat_dim, d_model=16, nhead=1, num_layers=1, dim_feedforward=64)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
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
        avg_train_loss = total_loss / num_batches
        
        # Validation
        transformer.eval()
        with torch.no_grad():
            val_output = transformer(val_received)
            val_loss = criterion(val_output.view(-1, 4), val_labels.view(-1)).item()
            predicted_classes = torch.argmax(val_output, dim=-1).view(-1).numpy()
            constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            equalized_complex = constellation[predicted_classes]
            val_ber = np.mean(qpsk_demodulate(equalized_complex) != val_bits.reshape(-1))
            val_ser = np.mean(predicted_classes != val_labels.view(-1).numpy())
        
        metrics[fd]['train_loss'].append(avg_train_loss)
        metrics[fd]['val_loss'].append(val_loss)
        metrics[fd]['val_ber'].append(val_ber)
        metrics[fd]['val_ser'].append(val_ser)
        
        if epoch % 10 == 0 or epoch == CONFIG['num_epochs'] - 1:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Val BER: {val_ber:.6f}, Val SER: {val_ser:.6f}")
    
    return transformer

# Main execution
for fd in CONFIG['fd_list']:
    train_data, val_data, test_data[fd], feat_dim = prepare_data(fd)
    transformer = train_model(train_data, val_data, feat_dim)
    test_data[fd]['transformer'] = transformer

# Evaluate test set
print("\n=== Test Set Evaluation ===")
test_metrics = evaluate_test_set(
    CONFIG['fd_list'], test_data, plots_dir, current_date,
    CONFIG['num_epochs'], metrics
)

# Print summary
print("\n=== Summary of Metrics ===")
for fd in CONFIG['fd_list']:
    print(f"\nDoppler Frequency: {fd} Hz")
    print("Epoch | Train Loss | Val Loss | Val BER | Val SER")
    print("-" * 50)
    for epoch in range(0, CONFIG['num_epochs'], 10):
        if epoch < len(metrics[fd]['train_loss']):
            print(f"{epoch:4d} | {metrics[fd]['train_loss'][epoch]:.6f} | "
                  f"{metrics[fd]['val_loss'][epoch]:.6f} | {metrics[fd]['val_ber'][epoch]:.6f} | "
                  f"{metrics[fd]['val_ser'][epoch]:.6f}")
    print(f"Final Test BER: {test_metrics[fd]['test_ber']:.6f}")
    print(f"Final Test SER: {test_metrics[fd]['test_ser']:.6f}")

print(f"\nSimulation completed! SNR: {CONFIG['snr_db']} dB, Symbols: {CONFIG['num_symbols']}")