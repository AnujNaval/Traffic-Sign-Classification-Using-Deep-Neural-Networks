# Traffic-Sign-Classification-Using-Deep-Neural-Networks

---

## 🔍 Features Implemented

### ✔ Generic Neural Network Framework
- Supports:
  - Any number of hidden layers
  - Any number of units per layer
  - Sigmoid or ReLU activation
  - Softmax output layer
  - Mini-batch SGD
  - Cross-entropy loss
  - Backpropagation implemented manually
  
### ✔ Experiments Implemented (as per assignment parts b–e)
- **Part (b):** Vary number of hidden units  
- **Part (c):** Vary network depth  
- **Part (d):** Adaptive learning rate ηₑ = η₀ / √e  
- **Part (e):** ReLU activation instead of sigmoid  

### ✔ Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Plots for average F1 vs number of units / depth  

---

## ▶️ How to Run

The script follows the **auto-evaluation format**:

### **Command**
python neural_network.py <train_data_path> <test_data_path> <output_folder_path> <question_part>


### **Arguments**
| Argument | Description |
|---------|-------------|
| `<train_data_path>` | Path to training CSV |
| `<test_data_path>` | Path to test CSV |
| `<output_folder_path>` | Folder where predictions will be saved |
| `<question_part>` | One of `b`, `c`, `d`, `e` |

### **Example**

python neural_network.py ./data/train.csv ./data/test.csv ./outputs b


After running, a file named:

prediction_<question_part>.csv


will be generated in the output folder.

Format of prediction file:

prediction
7
12
32
...

(One column, preserving test example order.)

---

## ⚙️ Hyperparameters Used

| Parameter | Value |
|----------|--------|
| Batch size (M) | 32 |
| Input dimension (n) | 2352 |
| Classes (r) | 43 |
| Learning rate | 0.01 (fixed) |
| Adaptive learning rate | ηₑ = η₀ / √e |
| Activation (default) | Sigmoid |
| Activation (part e) | ReLU |

---

## 📊 Outputs & Plots

The script produces:
- Test set predictions  
- Plots for  
  - Avg F1 vs hidden units (Part b)  
  - Avg F1 vs depth (Part c, d, e)  
- Performance metrics printed in the console  

---

## 🧠 Part-Wise Overview

### **Part (b): Single hidden layer experiments**
Hidden layer sizes: `{1, 5, 10, 50, 100}`  
Outputs: Precision/Recall/F1 per class + Avg F1 plot  

### **Part (c): Deeper architectures**
Architectures:
- `[512]`
- `[512, 256]`
- `[512, 256, 128]`
- `[512, 256, 128, 64]`

### **Part (d): Adaptive learning rate**
Learning rate decreases per epoch.  
Same architectures as part (c).  

### **Part (e): ReLU activation**
Same as part (d), but hidden layers use ReLU.  

---

## 📦 Requirements

Install dependencies:

pip install numpy pandas matplotlib scikit-learn


---

## ✍️ Notes
- All backpropagation and forward-pass computations are implemented **from scratch**.
- No deep learning libraries (PyTorch, TensorFlow, Keras) were used.
- Scikit-learn is used **only for evaluation metrics**.

---

## 🧑‍💻 Author
This project was implemented as part of **COL774: Machine Learning** coursework.

