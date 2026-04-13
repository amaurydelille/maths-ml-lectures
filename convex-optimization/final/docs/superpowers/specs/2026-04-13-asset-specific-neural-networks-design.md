# Asset-Specific Neural Networks for Portfolio Optimization

## Overview

This specification describes the implementation of asset-specific neural networks to replace the current Ridge regression approach for special (7-10) and mixed (4-6) assets in the portfolio optimization system. The goal is to maximize Sharpe ratio on test data while maintaining simplicity, interpretability, and fast training.

## Motivation

The current system uses Ridge regression for all asset groups, but different assets exhibit different signal characteristics:
- Linear assets (1-3): Simple linear relationships
- Mixed assets (4-6): Complex feature interactions
- Special assets (7-10): Non-linear patterns requiring more modeling capacity
- Noise assets (11-12): No signal, kept at zero weights

Neural networks can potentially capture non-linear patterns and feature interactions that Ridge regression misses, leading to better return predictions and higher Sharpe ratios.

## Architecture Design

### Asset-Specific Network Architectures

**Linear Assets (1-3)**: 
- Architecture: 20 → 12 → 3
- Rationale: Minimal complexity since these assets are already well-modeled by linear methods
- Activation: ReLU

**Mixed Assets (4-6)**:
- Architecture: 20 → 20 → 3  
- Rationale: Equal hidden layer size to capture feature interactions without over-parameterization
- Activation: ReLU

**Special Assets (7-10)**:
- Architecture: 20 → 16 → 4
- Rationale: Medium complexity for non-linear patterns
- Activation: ReLU

**Noise Assets (11-12)**:
- No network, weights set to zero as in current implementation

### Network Implementation

Each network is a simple 2-layer feedforward neural network:
```
Input (20 features) → Hidden Layer (ReLU) → Output Layer (linear)
```

All networks use the same training procedure but are optimized independently for their respective asset groups.

## Training Strategy

### Data Preprocessing
- Feature normalization: Z-score standardization using training mean/std
- Target normalization: Z-score standardization with clipping at ±5σ to handle outliers
- Same preprocessing pipeline as current Ridge implementation

### Training Procedure
- **Optimizer**: Simple SGD with fixed learning rate (no momentum)
- **Learning Rate**: 0.001 (fixed, no tuning required)
- **Batch Size**: Full batch (entire training set)
- **Epochs**: 150 epochs with early stopping
- **Early Stopping**: Monitor validation loss, stop if no improvement for 20 epochs
- **Validation Split**: 90/10 split of training data (since true validation set is hidden)

### Loss Function
**Huber Loss** with δ = 0.01:
```
L(r) = {
    0.5 * r²                    if |r| ≤ δ  
    δ * (|r| - 0.5 * δ)        if |r| > δ
}
```
Plus L2 regularization: `λ * ||W||²` with λ = 0.001

Huber loss provides robustness to outliers while maintaining smooth gradients, crucial for financial return prediction.

## Implementation Structure

### File Organization
- **New File**: `neural_model.py` - Contains all neural network implementation
- **Interface**: Drop-in replacement for `PortfolioChallengeModel`
- **Output**: Same submission.parquet format as current system

### Core Classes

**SimpleNetwork Class**:
```python
class SimpleNetwork:
    def __init__(self, input_size, hidden_size, output_size)
    def forward(self, X)           # Forward pass with ReLU activation
    def huber_loss(self, y_pred, y_true, delta=0.01)  # Huber loss computation
    def train(self, X, y, epochs=150, lr=0.001)       # SGD training with early stopping
```

**NeuralPortfolioModel Class**:
```python  
class NeuralPortfolioModel:
    def __init__(self)             # Initialize 3 networks for different asset groups
    def load_data(self, data_dir)  # Same interface as current model
    def fit(self, x_train, r_train) # Train all networks independently
    def predict_returns(self, x_df) # Generate predictions using trained networks
    def build_weights(self, pred_df) # Same portfolio construction as current
    def build_submission(self, x_train, x_test) # Same submission format
    def save_submission(self, submission, path) # Same saving mechanism
```

### Integration Points
- **Data Loading**: Reuse existing parquet loading infrastructure
- **Feature Engineering**: Same 20-feature input as current system  
- **Portfolio Construction**: Identical inverse volatility weighting and normalization
- **Output Format**: Exact same submission.parquet structure with pred_* and weight_* columns

## Error Handling & Robustness

### Numerical Stability
- Weight initialization: Xavier/Glorot initialization for stable gradients
- Gradient clipping: Clip gradients to [-1, 1] to prevent exploding gradients
- NaN handling: Check for NaN values after each forward pass, reinitialize if detected

### Overfitting Prevention  
- Early stopping based on validation loss
- L2 regularization on all weights
- Simple architectures to reduce parameter count
- Fixed hyperparameters to avoid overfitting to validation performance

### Fallback Strategy
If any network fails to train properly (loss doesn't decrease), fall back to Ridge regression for that asset group to ensure robust predictions.

## Expected Outcomes

### Performance Targets
- **Training Speed**: <30 seconds total training time for all networks
- **Memory Usage**: Minimal - small networks with <1000 parameters each
- **Interpretability**: Clear asset group separation, visualizable weight patterns
- **Sharpe Improvement**: Target >10% improvement over current Ridge approach

### Success Metrics
- Test Sharpe ratio higher than current Ridge baseline
- Stable training (loss consistently decreases)
- No significant overfitting (train vs validation loss gap <20%)
- Fast inference time for submission generation

## Testing Strategy

### Validation Approach
- Use internal 90/10 split for early stopping and architecture validation
- Compare neural network predictions vs Ridge baseline on same data splits
- Monitor training curves to ensure stable convergence

### Performance Comparison
- A/B test: Run both neural networks and Ridge on same data
- Compare Sharpe ratios, MSE, and prediction stability
- Ensure neural approach consistently outperforms before deployment

## Risk Mitigation

### Overfitting Risks
- Simple architectures reduce overfitting potential
- Fixed hyperparameters prevent validation set snooping
- Early stopping prevents training too long
- Regularization adds generalization pressure

### Implementation Risks  
- Fallback to Ridge if neural training fails
- Extensive input validation and error handling
- Modular design allows easy debugging and replacement

### Computational Risks
- Fixed epoch limits prevent infinite training
- Memory-efficient implementation for large datasets
- Batch processing for inference if needed

This design balances the potential benefits of neural networks (non-linear modeling, automatic feature interactions) with the constraints of simplicity, interpretability, and robust performance required for the portfolio optimization challenge.