import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import qpsk_demodulate

def evaluate_test_set(fd_list, test_data, plots_dir, current_date, num_epochs, metrics):
    """Evaluate the transformer model on the test set and generate plots."""
    test_metrics = {fd: {'test_ber': None, 'test_ser': None} for fd in fd_list}
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    for fd in fd_list:
        print(f"\nEvaluating test set for fd={fd} Hz...")
        transformer = test_data[fd]['transformer']
        test_received = test_data[fd]['received']
        test_labels = test_data[fd]['labels']
        test_bits = test_data[fd]['bits']
        test_transmitted = test_data[fd]['transmitted']
        received_baseband = test_data[fd].get('received_baseband', test_transmitted)
        
        transformer.eval()
        with torch.no_grad():
            test_output = transformer(test_received)
            predicted_classes = torch.argmax(test_output, dim=-1)
            predicted_classes_flat = predicted_classes.view(-1).numpy()
            equalized_complex = constellation[predicted_classes_flat]
            test_classes_flat = test_labels.view(-1).numpy()
            test_metrics[fd]['test_ser'] = np.mean(predicted_classes_flat != test_classes_flat)
            test_metrics[fd]['test_ber'] = np.mean(qpsk_demodulate(equalized_complex) != test_bits.reshape(-1))
        
        # Generate constellation plot
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.scatter(np.real(test_transmitted[:200]), np.imag(test_transmitted[:200]), alpha=0.6, s=20)
        plt.title(f'Transmitted Symbols (fd={fd} Hz)')
        plt.grid(True)
        plt.axis('equal')
        plt.subplot(1, 3, 2)
        plt.scatter(np.real(received_baseband[:200]), np.imag(received_baseband[:200]), alpha=0.6, s=20)
        plt.title(f'Received Symbols (fd={fd} Hz)')
        plt.grid(True)
        plt.axis('equal')
        plt.subplot(1, 3, 3)
        plt.scatter(np.real(equalized_complex[:200]), np.imag(equalized_complex[:200]), alpha=0.6, s=20)
        plt.title(f'Equalized Symbols (fd={fd} Hz)')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/constellation_fd_{fd}_{current_date}.png")
        plt.close()
        
        # Generate BER/SER plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(num_epochs), metrics[fd]['val_ber'], label='Validation BER')
        plt.title(f'BER vs Epoch (fd={fd} Hz)')
        plt.xlabel('Epoch')
        plt.ylabel('BER')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(num_epochs), metrics[fd]['val_ser'], label='Validation SER')
        plt.title(f'SER vs Epoch (fd={fd} Hz)')
        plt.xlabel('Epoch')
        plt.ylabel('SER')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/ber_ser_fd_{fd}_{current_date}.png")
        plt.close()
    
    # Plot BER/SER vs Doppler Frequency
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fd_list, [test_metrics[fd]['test_ber'] for fd in fd_list], marker='o')
    plt.title('Test BER vs Doppler Frequency')
    plt.xlabel('Doppler Frequency (Hz)')
    plt.ylabel('BER')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(fd_list, [test_metrics[fd]['test_ser'] for fd in fd_list], marker='o')
    plt.title('Test SER vs Doppler Frequency')
    plt.xlabel('Doppler Frequency (Hz)')
    plt.ylabel('SER')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/ber_ser_vs_doppler_{current_date}.png")
    plt.close()
    
    # Print test metrics summary
    print("\n=== Test Metrics Summary ===")
    print("Doppler Frequency (Hz) | Test BER | Test SER")
    print("-" * 40)
    for fd in fd_list:
        print(f"{fd:>20} | {test_metrics[fd]['test_ber']:.6f} | {test_metrics[fd]['test_ser']:.6f}")
    
    return test_metrics