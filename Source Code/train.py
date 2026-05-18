"""
Main Training Script
=====================
Project    : Artificial Intelligence for Real-Time ECG Analysis and Smart Cardiac Monitoring
Author     : Shubham Guglani (Roll No: 2210990846)
Supervisor : Dr. Preeti Saini
Dept       : CSE, Chitkara University

Usage:
    # Train on synthetic data (demo):
    python train.py --data synthetic --epochs 30

    # Train on MIT-BIH:
    python train.py --data mitbih --data_path ./data/mitbih --epochs 50
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from data.dataset_loader import load_synthetic, load_mitbih, MITBIH_CLASS_MAP
from models.cnn_bilstm import build_cnn_bilstm, train_model
from models.compression import prune_weights


# ─── Argument Parser ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='ECG Arrhythmia Classification Training')
    parser.add_argument('--data',       type=str, default='synthetic',
                        choices=['synthetic', 'mitbih'],
                        help='Dataset to use for training')
    parser.add_argument('--data_path',  type=str, default='./data/mitbih',
                        help='Path to MIT-BIH data directory')
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--prune',      action='store_true',
                        help='Apply weight pruning after training')
    parser.add_argument('--quantize',   action='store_true',
                        help='Quantize model to INT8 TFLite after training')
    parser.add_argument('--output_dir', type=str, default='./models')
    return parser.parse_args()


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, class_names, output_dir):
    """Run evaluation and save confusion matrix + classification report."""
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=list(class_names.keys()))
    print("\n── Classification Report ──────────────────────────────────────")
    print(report)

    # Save report to file
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("ECG Arrhythmia Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(class_names.keys()),
                yticklabels=list(class_names.keys()))
    plt.title('ECG Arrhythmia Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()

    # Overall accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
    return accuracy


# ─── Training History Plot ────────────────────────────────────────────────────
def plot_training_history(history, output_dir):
    """Plot and save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Training history plot saved to: {plot_path}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("ECG Arrhythmia Classification - Training Pipeline")
    print("Author: Shubham Guglani | Roll No: 2210990846")
    print("Supervisor: Dr. Preeti Saini | Chitkara University")
    print("=" * 60)

    # 1. Load dataset
    print(f"\n[1/5] Loading dataset: {args.data}")
    if args.data == 'synthetic':
        data = load_synthetic(n_samples_per_class=1000)
    else:
        data = load_mitbih(args.data_path)

    X_train, X_test  = data['X_train'], data['X_test']
    y_train, y_test  = data['y_train'], data['y_test']
    class_names      = data['classes']

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # 2. Build model
    print("\n[2/5] Building CNN-BiLSTM model...")
    model = build_cnn_bilstm(
        input_length=X_train.shape[1],
        num_classes=y_train.shape[1]
    )
    model.summary()

    # 3. Train
    print(f"\n[3/5] Training for up to {args.epochs} epochs...")
    save_path = os.path.join(args.output_dir, 'ecg_cnn_bilstm.h5')
    history   = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=save_path
    )
    plot_training_history(history, args.output_dir)

    # 4. Evaluate
    print("\n[4/5] Evaluating model...")
    evaluate_model(model, X_test, y_test, class_names, args.output_dir)

    # 5. Optional compression
    if args.prune:
        print("\n[5/5] Applying weight pruning (50% sparsity)...")
        model = prune_weights(model, sparsity=0.5)
        pruned_path = os.path.join(args.output_dir, 'ecg_cnn_bilstm_pruned.h5')
        model.save(pruned_path)
        print(f"Pruned model saved to: {pruned_path}")

    if args.quantize:
        print("\n[5/5] Quantizing model to INT8 TFLite...")
        from models.compression import quantize_model, make_representative_dataset
        tflite_path = os.path.join(args.output_dir, 'ecg_quantized.tflite')
        rep_dataset = make_representative_dataset(X_train)
        quantize_model(model, rep_dataset, tflite_path)

    print("\n✅ Training pipeline complete!")
    print(f"   All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
