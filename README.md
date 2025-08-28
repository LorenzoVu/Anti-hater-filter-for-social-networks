# Anti-Hater Filter for Social Networks

## Project Overview
This project implements a multi-label toxicity classifier designed to automatically moderate user comments across 6 categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. The model is optimized to reduce manual moderation load while maintaining high-quality discussion, with a focus on maximizing true positives and minimizing false negatives for rare and critical toxicity classes.

## Problem Description
- **Task**: Multi-label text classification for toxicity detection
- **Dataset**: ~160k user comments with binary multi-label annotations
- **Challenge**: Strong class imbalance (e.g., `threat` ~0.3%, `identity_hate` <1%)
- **Goal**: The model outputs 6 probability scores (one per category) with optimized classification thresholds

## Model Architecture
- **Text Processing**: 
  - Normalization (lowercase, emoji conversion, de-elongation, leet-to-plain text)
  - Word-level tokenization with 20k vocabulary size
  - Max sequence length of 128 tokens

- **Neural Network**:
  - Embedding layer (200 dimensions) with `mask_zero=True`
  - Spatial Dropout (0.2)
  - 2x Bidirectional LSTM layers with recurrent dropout
  - Mask-aware pooling (Global Max + Global Average)
  - Dense layer (128 units, ReLU activation)
  - Output layer (6 units, sigmoid activation)

- **Training Approach**:
  - Weighted Binary Cross-Entropy loss to handle class imbalance
  - AdamW optimizer with learning rate 1e-3 and weight decay
  - Monitoring PR-AUC for callbacks and validation
  - Per-class threshold optimization from Precision-Recall curves

## Performance Highlights
- **Micro F1**: 0.67
- **Macro F1**: 0.53
- **Class-specific PR-AUC**: Values well above prevalence for all classes
- **Threshold Tuning**: Precision-floor strategy for optimal recall-precision tradeoff

## Files Description
- `filter.ipynb`: Main notebook with all code and explanations
- `*.keras`: Saved model files (best and final versions)
- `*.npy`: Preprocessed data (tokenized inputs, labels, thresholds)
- `vocabulary.txt`: Saved vocabulary from text vectorization
- `class_weights.json`: Computed class weights for handling imbalance
- `training_history.pkl`: Saved training metrics history

## How to Use
1. **Setup**: Install required packages:
   ```
   pip install pandas numpy matplotlib seaborn regex emoji unidecode tqdm tensorflow scikit-learn
   ```

2. **Preprocessing**: The notebook contains all preprocessing steps:
   - Data loading and exploration
   - Text normalization and tokenization
   - Stratified train/val/test splitting

3. **Training**: The model training section includes:
   - Class weight calculation
   - Model definition with custom pooling layers
   - Training with appropriate callbacks

4. **Evaluation**: The evaluation includes:
   - Threshold tuning methods (F-beta and precision-floor)
   - Comprehensive metrics (precision, recall, F1, TP/FP/FN counts)
   - PR-AUC per class

5. **Inference**: Use the saved model and thresholds for new data

## Conclusions
The model achieves strong performance on frequent toxicity classes and competitive results on rare ones. The threshold tuning approach allows for flexible operational trade-offs based on moderation policies:

- **Recall-first strategy** for critical classes like `threat` and `identity_hate`
- **Balanced precision-recall** for more common classes like `toxic` and `obscene`
- **Customizable thresholds** to adapt to different moderation needs

The model's strong ranking ability (high PR-AUC per class) ensures good operating points exist, and threshold tuning effectively rebalances the error profile according to policy needs.
