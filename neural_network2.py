# neural_network.py
import numpy as np
import os
import argparse
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Determine weight initialization based on activation
        layer_sizes = [input_size] + hidden_layers
        for i in range(len(layer_sizes) - 1):
            if self.activation == 'relu':
                # He initialization for ReLU
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]))
            else:
                # Xavier initialization for sigmoid
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1. / layer_sizes[i]))
            self.biases.append(np.zeros(layer_sizes[i+1]))
        
        # Output layer (always uses softmax)
        self.weights.append(np.random.randn(layer_sizes[-1], output_size) * np.sqrt(1. / layer_sizes[-1]))
        self.biases.append(np.zeros(output_size))
        
        # For early stopping
        self.best_weights = None
        self.best_biases = None
        self.best_loss = float('inf')
        self.patience_counter = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        
        # Hidden layers
        for i in range(len(self.hidden_layers)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            if self.activation == 'relu':
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            self.zs.append(z)
            self.activations.append(a)
        
        # Output layer
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a_out = self.softmax(z_out)
        self.zs.append(z_out)
        self.activations.append(a_out)
        return a_out

    def compute_loss(self, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(self.activations[-1][range(m), y_true])
        return np.sum(log_likelihood) / m

    def backward(self, X, y_true):
        m = y_true.shape[0]
        delta = self.activations[-1].copy()
        delta[range(m), y_true] -= 1
        delta /= m
        deltas = [delta]
        
        # Backpropagate through hidden layers
        for i in reversed(range(len(self.hidden_layers))):
            if self.activation == 'relu':
                derv = self.relu_derivative(self.zs[i])
            else:
                derv = self.sigmoid_derivative(self.activations[i+1])
            delta_current = np.dot(deltas[-1], self.weights[i+1].T) * derv
            deltas.append(delta_current)
        
        # Reverse to get correct order
        deltas.reverse()
        
        # Update weights and biases
        for i in range(len(self.weights)):
            grad_w = np.dot(self.activations[i].T, deltas[i])
            grad_b = np.sum(deltas[i], axis=0)
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    def fit(self, X, y, epochs=50, batch_size=32, patience=10, min_delta=0.0001, adaptive_lr=True):
        initial_lr = self.learning_rate
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        for epoch in range(epochs):
            if adaptive_lr:
                # Update learning rate for current epoch
                self.learning_rate = initial_lr / np.sqrt(epoch + 1)
            
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            epoch_loss = 0
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                self.forward(X_batch)
                loss = self.compute_loss(y_batch)
                epoch_loss += loss
                self.backward(X_batch, y_batch)
            
            avg_loss = epoch_loss / X.shape[0]
            
            # Early stopping logic
            if self.best_loss - avg_loss > min_delta:
                self.best_loss = avg_loss
                self.patience_counter = 0
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.weights = [w.copy() for w in self.best_weights]
                    self.biases = [b.copy() for b in self.best_biases]
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Learning Rate: {self.learning_rate:.6f}")

        # Restore initial learning rate
        self.learning_rate = initial_lr

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

def load_images_from_folder(folder):
    images = []
    labels = []
    for class_dir in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, class_dir)
        if os.path.isdir(class_path):
            for img_file in sorted(os.listdir(class_path)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (28, 28))
                        images.append(img.flatten())
                        labels.append(int(class_dir))
    return np.array(images), np.array(labels)

def load_test_data(test_dir):
    images = []
    filenames = []
    for img_file in sorted(os.listdir(test_dir)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (28, 28))
                images.append(img.flatten())
                filenames.append(img_file)
    return np.array(images), filenames

def load_test_labels():
    df = pd.read_csv("test_labels.csv")
    for col in ['ClassId', 'classid', 'Class', 'class', 'Label', 'label', 'target']:
        if col in df.columns:
            return df[col].values
    raise KeyError("Class ID column not found")

def part_a(train_path, test_path, output_folder):
    print("Loading training data...")
    X_train, y_train = load_images_from_folder(train_path)
    print("Loading test data...")
    X_test, filenames = load_test_data(test_path)
    print("Loading test labels...")
    y_test = load_test_labels()

    # Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train / 255.0)
    X_test = scaler.transform(X_test / 255.0)

    # Initialize network with ReLU
    nn = NeuralNetwork(
        input_size=2352,
        hidden_layers=[512],
        output_size=43,
        learning_rate=0.001,
        activation='relu'
    )
    
    # Train with early stopping
    nn.fit(X_train, y_train, epochs=200, batch_size=32, patience=15)
    
    # Predictions
    y_pred = nn.predict(X_test)
    
    # Save results
    pd.DataFrame({'filename': filenames, 'prediction': y_pred}).to_csv(
        os.path.join(output_folder, 'prediction_a.csv'), index=False)
    
    # Metrics
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

def part_b(train_path, test_path, output_folder):
    print("Loading training data...")
    X_train, y_train = load_images_from_folder(train_path)
    print("Loading test data...")
    X_test, filenames = load_test_data(test_path)
    print("Loading test labels...")
    y_test = load_test_labels()

    # Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train / 255.0)
    X_test = scaler.transform(X_test / 255.0)

    hidden_units = [1, 5, 10, 50, 100]
    train_f1 = []
    test_f1 = []

    for units in hidden_units:
        print(f"\nTraining with {units} units...")
        nn = NeuralNetwork(
            input_size=2352,
            hidden_layers=[units],
            output_size=43,
            learning_rate=0.01,
            activation='sigmoid'
        )
        nn.fit(X_train, y_train, epochs=100, batch_size=32)
        
        # Metrics
        y_train_pred = nn.predict(X_train)
        train_f1.append(f1_score(y_train, y_train_pred, average='macro'))
        y_test_pred = nn.predict(X_test)
        test_f1.append(f1_score(y_test, y_test_pred, average='macro'))
        
        # Save predictions
        pd.DataFrame({'prediction': y_test_pred}).to_csv(
            os.path.join(output_folder, f'prediction_b_{units}.csv'), index=False)

    # Plot
    plt.figure()
    plt.plot(hidden_units, train_f1, label='Train F1')
    plt.plot(hidden_units, test_f1, label='Test F1')
    plt.xlabel('Hidden Units')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'part_b_f1.png'))
    plt.close()

    # Save best model
    best_idx = np.argmax(test_f1)
    best_units = hidden_units[best_idx]
    print(f"Best model: {best_units} units")
    pd.DataFrame({'prediction': nn.predict(X_test)}).to_csv(
        os.path.join(output_folder, 'prediction_b.csv'), index=False)

def part_c(train_path, test_path, output_folder):
    print("Loading training data...")
    X_train, y_train = load_images_from_folder(train_path)
    print("Loading test data...")
    X_test, filenames = load_test_data(test_path)
    print("Loading test labels...")
    y_test = load_test_labels()

    # Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train / 255.0)
    X_test = scaler.transform(X_test / 255.0)

    architectures = [
        [512],
        [512, 256],
        [512, 256, 128],
        [512, 256, 128, 64]
    ]
    
    train_f1_scores = []
    test_f1_scores = []
    depths = []

    for arch in architectures:
        print(f"\nTraining architecture: {arch}")
        nn = NeuralNetwork(
            input_size=2352,
            hidden_layers=arch,
            output_size=43,
            learning_rate=0.01,
            activation='sigmoid'
        )
        nn.fit(X_train, y_train, epochs=100, batch_size=32)
        
        # Train metrics
        y_train_pred = nn.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        train_f1_scores.append(train_f1)
        
        # Test metrics
        y_test_pred = nn.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        test_f1_scores.append(test_f1)
        
        depths.append(len(arch))
        
        # Save predictions
        pd.DataFrame({'prediction': y_test_pred}).to_csv(
            os.path.join(output_folder, f'prediction_c_{"_".join(map(str, arch))}.csv'),
            index=False
        )

    # Plot F1 vs depth
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_f1_scores, marker='o', label='Train F1')
    plt.plot(depths, test_f1_scores, marker='o', label='Test F1')
    plt.xlabel('Network Depth')
    plt.ylabel('Macro F1 Score')
    plt.xticks(depths)
    plt.title('F1 Score vs Network Depth')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'part_c_f1_depth.png'))
    plt.close()

def part_d(train_path, test_path, output_folder):
    print("Loading training data...")
    X_train, y_train = load_images_from_folder(train_path)
    print("Loading test data...")
    X_test, filenames = load_test_data(test_path)
    print("Loading test labels...")
    y_test = load_test_labels()

    # Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train / 255.0)
    X_test = scaler.transform(X_test / 255.0)

    architectures = [
        [512],
        [512, 256],
        [512, 256, 128],
        [512, 256, 128, 64]
    ]
    
    train_f1_scores = []
    test_f1_scores = []
    depths = []

    for arch in architectures:
        print(f"\nTraining architecture: {arch} with adaptive learning rate")
        nn = NeuralNetwork(
            input_size=2352,
            hidden_layers=arch,
            output_size=43,
            learning_rate=0.01,
            activation='sigmoid'
        )
        nn.fit(X_train, y_train, epochs=50, batch_size=32, patience=10, adaptive_lr=True)
        
        # Metrics
        y_train_pred = nn.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        train_f1_scores.append(train_f1)
        
        y_test_pred = nn.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        test_f1_scores.append(test_f1)
        
        depths.append(len(arch))
        
        # Save predictions
        pd.DataFrame({'prediction': y_test_pred}).to_csv(
            os.path.join(output_folder, f'prediction_d_{"_".join(map(str, arch))}.csv'),
            index=False
        )

    # Plot F1 vs depth
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_f1_scores, marker='o', label='Train F1')
    plt.plot(depths, test_f1_scores, marker='o', label='Test F1')
    plt.xlabel('Network Depth')
    plt.ylabel('Macro F1 Score')
    plt.xticks(depths)
    plt.title('F1 Score vs Network Depth with Adaptive Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'part_d_f1_depth.png'))
    plt.close()

def part_e(train_path, test_path, output_folder):
    print("Loading training data...")
    X_train, y_train = load_images_from_folder(train_path)
    print("Loading test data...")
    X_test, filenames = load_test_data(test_path)
    print("Loading test labels...")
    y_test = load_test_labels()

    # Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train / 255.0)
    X_test = scaler.transform(X_test / 255.0)

    architectures = [
        [512],
        [512, 256],
        [512, 256, 128],
        [512, 256, 128, 64]
    ]
    
    train_f1_scores = []
    test_f1_scores = []
    depths = []

    for arch in architectures:
        print(f"\nTraining architecture: {arch} with ReLU activation")
        nn = NeuralNetwork(
            input_size=2352,
            hidden_layers=arch,
            output_size=43,
            learning_rate=0.01,
            activation='relu'
        )
        nn.fit(X_train, y_train, epochs=50, batch_size=32, patience=10, adaptive_lr=True)
        
        # Metrics
        y_train_pred = nn.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        train_f1_scores.append(train_f1)
        
        y_test_pred = nn.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        test_f1_scores.append(test_f1)
        
        depths.append(len(arch))
        
        # Save predictions
        pd.DataFrame({'prediction': y_test_pred}).to_csv(
            os.path.join(output_folder, f'prediction_e_{"_".join(map(str, arch))}.csv'),
            index=False
        )

    # Plot F1 vs depth
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_f1_scores, marker='o', label='Train F1')
    plt.plot(depths, test_f1_scores, marker='o', label='Test F1')
    plt.xlabel('Network Depth')
    plt.ylabel('Macro F1 Score')
    plt.xticks(depths)
    plt.title('F1 Score vs Network Depth with ReLU Activation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'part_e_f1_depth.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Neural Network for GTSRB')
    parser.add_argument('train_data_path', help='Path to training data directory')
    parser.add_argument('test_data_path', help='Path to test data directory')
    parser.add_argument('output_folder_path', help='Path to output folder')
    parser.add_argument('question_part', help='Question part (a, b, c, d, or e)')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    if args.question_part == 'a':
        part_a(args.train_data_path, args.test_data_path, args.output_folder_path)
    elif args.question_part == 'b':
        part_b(args.train_data_path, args.test_data_path, args.output_folder_path)
    elif args.question_part == 'c':
        part_c(args.train_data_path, args.test_data_path, args.output_folder_path)
    elif args.question_part == 'd':
        part_d(args.train_data_path, args.test_data_path, args.output_folder_path)
    elif args.question_part == 'e':
        part_e(args.train_data_path, args.test_data_path, args.output_folder_path)
    else:
        print("Invalid question part. Please use a, b, c, d, or e.")

if __name__ == '__main__':
    main()