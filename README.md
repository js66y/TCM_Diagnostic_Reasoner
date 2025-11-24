# TCDR: Topology-Constrained Diagnostic Reasoner

This repository contains the source code and implementation for the paper:

**"Establishing a Large-Scale TCM EHR Benchmark with a Knowledge-Guided Dual-Branch Diagnostic Reasoner"**

This project utilizes a knowledge-guided deep learning approach to recommend Traditional Chinese Medicine (TCM) herbs. It integrates multi-level information including **Preliminary Symptoms**, **TCM Symptoms**, **Syndrome Patterns**, and **Methods** to perform accurate herb recommendation and diagnostic reasoning.

## 📂 Project Structure

```text
TCM_copy/
├── dataloader/          # Data loading logic
├── dataset/             # Dataset files
│   ├── binary_matrices_graph/ # Adjacency matrices for graph structures
│   ├── binary_matrices_mask/  # Matrices used for masking mechanisms
│   ├── map/                   # Mappings between IDs and Text (.pkl)
│   ├── train.pkl / test.pkl   # Training and testing splits
│   └── ...
├── logs/                # Logs and saved model checkpoints
├── modules/             # Core neural network modules
├── utils/               # Utility functions
├── main_batch.py        # Main entry script (Train, Test, Predict)
├── parse.py             # Argument parsing and configuration
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

## 🛠️ Installation

Please ensure that Python 3.8+ is installed. It is recommended to run this project in a virtual environment.

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

The main entry point for the project is `main_batch.py`. You can switch between training, testing, and prediction modes using command-line arguments.

### 1. Training

To train the model from scratch:

```bash
python main_batch.py --train --model_name TCM --version tcdr --epochs 50 --batch_size 64
```

*Note: Logs and model checkpoints will be saved in the `logs/` directory by default.*

### 2. Testing

To evaluate a pre-trained model on the test set:

```bash
# Replace 'path_to_checkpoint.pt' with your actual model path
python main_batch.py --test --pre_train "logs/n-TCM/v-tcdr/your_timestamp/Best_state.pt"
```

### 3. Prediction

To run a single prediction task:

```bash
python main_batch.py --predict --pre_train "logs/n-TCM/v-tcdr/your_timestamp/Best_state.pt"
```

## ⚙️ Arguments

Key arguments defined in `parse.py`:

| Argument       | Default        | Description                           |
|:-------------- |:-------------- |:------------------------------------- |
| `--model_name` | `TCM`          | Name of the model                     |
| `--version`    | `tcdr`         | Version tag for the experiment        |
| `--data_dir`   | `.../dataset/` | Root directory of the dataset         |
| `--batch_size` | `64`           | Batch size for training/testing       |
| `--dim`        | `64`           | Dimension of hidden layers            |
| `--lr`         | `3e-4`         | Learning rate                         |
| `--epochs`     | `3`            | Number of training epochs             |
| `--pre_train`  | `None`         | Path to the pre-trained model (`.pt`) |
| `--device`     | `1`            | GPU Device ID                         |
| `--wandb`      | `False`        | Enable Weights & Biases logging       |

> **Note:** Please check the default paths in `parse.py`. If your data is stored in a different location, ensure you pass the correct path via the `--data_dir` argument or modify the `parse.py` file directly.

## 🖼️ Model Architecture

The overall architecture of the proposed model is shown below:

![Model Architecture](model.png)

## 📈 Experimental Results

An example of the model's performance on the test set:

![Test Results](test.jpg)
