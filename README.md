🛡️ Hybrid Graph Transformer for Network Intrusion Detection System
📖 Overview

This repository implements a novel Hybrid Graph Transformer model for Network Intrusion Detection Systems (NIDS). The model combines Graph Neural Networks (GNNs) with Transformer attention mechanisms to effectively detect and classify network attacks in both binary and multiclass scenarios.

🎯 Key Features

Hybrid Architecture: Combines GNN spatial processing with Transformer sequential analysis

Temporal Attention: Captures time-dependent attack patterns

Self-Supervised Learning: Utilizes contrastive learning for unlabeled data

Multi-class & Binary Classification: Identifies both general and specific attack types

Real-time Capable: Optimized for efficient inference

📊 Performance
✅ Binary Classification Results (Our Model)

Accuracy: 99.67%

Precision: 0.94 (Attack), 1.00 (Normal)

Recall: 0.98 (Attack), 1.00 (Normal)

Weighted F1-Score: 0.9967

📈 Confusion Matrix

📊 Classification Report Heatmap

📉 True vs Predicted Distribution

🏗️ Architecture

The model consists of three main components:

Graph Feature Extractor: Learns spatial dependencies between network nodes

Transformer Encoder with Multi-Head Attention: Analyzes sequential attack patterns

Classifier Head: Performs binary/multiclass prediction

📚 Research Basis

This implementation is inspired by the research paper:
"Applying Self-supervised Learning to Network Intrusion Detection for Network Flows with Graph Neural Network"
by Renjie Xu et al.

🔑 Key Improvements

Incorporation of Transformer layers

Temporal attention mechanisms

Adaptive contrastive learning

Enhanced feature extraction

🚀 Installation

# Clone repository

git clone https://github.com/Madhav17777777/Intrusion-Detection-System.git
cd Intrusion-Detection-System

# Install dependencies

pip install -r requirements.txt

💻 Usage
🔹 Training
from models.hybrid_transformer import HybridGraphTransformer

# Initialize model

model = HybridGraphTransformer(num_features=43, num_classes=2)

# Train model

train_model(model, train_loader, optimizer, device)

🔹 Inference

# Load trained model

model.load_state_dict(torch.load('models/trained_model.pth'))

# Make predictions

predictions = model(test_data)

🔹 Evaluation

# Comprehensive evaluation

evaluate_model(model, test_loader, device)

# Generate plots

plot_results(true_labels, predictions)

📁 Dataset

We use the NF-UNSW-NB15-v2 dataset, which contains:

45 network flow features

Binary labels (Normal / Attack)

Multiclass attack categories

Over 700,000 network flow samples

🧪 Experiments
🔹 Binary Classification Results
Model Accuracy Precision Recall F1-Score
Hybrid Graph Transformer (Ours) 99.67% 99.42% 99.64% 99.67%
Random Forest 99.65% 99.20% 99.20% 99.05%
XGBoost 99.58% 98.75% 99.10% 98.92%
🔹 Multiclass Classification Results
Attack Type Precision Recall F1-Score
DDoS 98.2% 97.8% 98.0%
DoS 97.5% 96.9% 97.2%
Reconnaissance 96.8% 95.4% 96.1%
👥 Contributors

Madhav Goel

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

Inspired by the research of Renjie Xu et al.

Thanks to the authors of the NF-UNSW-NB15-v2 dataset

Built with PyTorch and open-source libraries

📞 Contact

For questions or collaborations, please contact:
📧 23UCS635@lnmiit.ac.in
