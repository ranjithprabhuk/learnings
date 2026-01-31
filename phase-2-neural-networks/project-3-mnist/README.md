# Project 3: MNIST Handwritten Digit Recognition
## Neural Network Image Classification with TensorFlow.js

---

## Overview

Build a deep neural network that recognizes handwritten digits (0-9) from the famous **MNIST dataset**. This is the "Hello World" of deep learning and computer vision.

**What You'll Build**:
- Feedforward neural network with TensorFlow.js
- Train on 60,000 handwritten digit images
- Achieve >95% accuracy
- Real-time digit prediction
- Visualization of model performance

**Time Estimate**: 8-12 hours

---

## Learning Objectives

By completing this project, you will:
- ✅ Build and train neural networks with TensorFlow.js
- ✅ Load and preprocess image datasets
- ✅ Understand forward and backward propagation in practice
- ✅ Visualize training progress (loss and accuracy curves)
- ✅ Evaluate model performance with confusion matrices
- ✅ Understand concepts like epochs, batches, and learning rates
- ✅ Save and load trained models

---

## Dataset: MNIST

**What is MNIST?**
- 70,000 grayscale images of handwritten digits
- 60,000 training images
- 10,000 test images
- Each image is 28×28 pixels
- 10 classes (digits 0-9)

**Why MNIST?**
- Perfect for learning (small, fast to train)
- Standard benchmark dataset
- Well-studied problem
- Can achieve high accuracy with simple models

**Sample Images**:
```
0: ⚪⚪⚪⚫⚫⚫⚪⚪    1: ⚪⚪⚫⚫⚪⚪⚪    2: ⚫⚫⚫⚫⚫⚪⚪
   ⚪⚪⚫⚫⚪⚫⚫⚪       ⚪⚫⚫⚫⚪⚪⚪       ⚪⚪⚪⚪⚫⚫⚪
   ⚪⚫⚪⚪⚪⚪⚫⚪       ⚪⚪⚫⚫⚪⚪⚪       ⚪⚪⚪⚫⚫⚪⚪
   ⚪⚫⚪⚪⚪⚪⚫⚪       ⚪⚪⚫⚫⚪⚪⚪       ⚪⚪⚫⚫⚪⚪⚪
   ⚪⚫⚪⚪⚪⚪⚫⚪       ⚪⚪⚫⚫⚪⚪⚪       ⚪⚫⚫⚪⚪⚪⚪
   ⚪⚪⚫⚫⚫⚫⚪⚪       ⚪⚪⚫⚫⚪⚪⚪       ⚫⚫⚫⚫⚫⚫⚫
```

---

## Architecture

### Model Design

**Network Structure**:
```
Input Layer: 784 neurons (28×28 flattened)
    ↓
Hidden Layer 1: 128 neurons + ReLU
    ↓
Dropout: 0.2 (prevents overfitting)
    ↓
Hidden Layer 2: 64 neurons + ReLU
    ↓
Dropout: 0.2
    ↓
Output Layer: 10 neurons + Softmax (one per digit)
```

**Why This Architecture?**

**Input (784 neurons)**:
- 28×28 image = 784 pixels
- Each pixel is one feature
- Flattened to 1D array

**Hidden Layer 1 (128 neurons)**:
- Learns low-level features (edges, curves)
- ReLU activation for non-linearity
- More neurons = more representational capacity

**Dropout (0.2)**:
- Randomly drops 20% of neurons during training
- Prevents overfitting
- Forces network to learn robust features

**Hidden Layer 2 (64 neurons)**:
- Learns higher-level features (digit shapes)
- Combines features from layer 1
- Smaller than layer 1 (funnel architecture)

**Output (10 neurons)**:
- One neuron per class (0-9)
- Softmax converts to probabilities
- Sum of all outputs = 1

### Hyperparameters

```typescript
const config = {
  learningRate: 0.001,      // Step size for gradient descent
  epochs: 10,               // Full passes through training data
  batchSize: 128,           // Samples per gradient update
  validationSplit: 0.15,    // 15% of training data for validation
  optimizer: 'adam',        // Adaptive optimizer (best default)
  loss: 'categoricalCrossentropy',  // For multi-class classification
};
```

**Why These Values?**

**Learning Rate (0.001)**:
- Small enough to be stable
- Large enough to train in reasonable time
- Standard default for Adam optimizer

**Epochs (10)**:
- Enough to reach good accuracy (>95%)
- Not too many (prevents overfitting)
- Can train more if needed

**Batch Size (128)**:
- Good balance between speed and memory
- Larger = faster but needs more memory
- Smaller = more gradient updates but slower

**Adam Optimizer**:
- Adapts learning rate automatically
- Combines momentum and RMSprop
- Best general-purpose optimizer

---

## Implementation

### Project Structure

```
project-3-mnist/
├── src/
│   ├── mnist-classifier.ts    # Main neural network class
│   ├── data-loader.ts          # Load and preprocess MNIST data
│   ├── utils.ts                # Utility functions (metrics, viz)
│   ├── demo.ts                 # Basic training demo
│   ├── demo-train.ts           # Full training pipeline
│   ├── demo-predict.ts         # Prediction examples
│   ├── demo-visualize.ts       # Visualize results
│   └── demo-evaluate.ts        # Detailed evaluation
├── models/                     # Saved models (gitignored)
├── data/                       # MNIST data (gitignored)
├── package.json
├── tsconfig.json
└── README.md
```

### Core Components

#### 1. **MNISTClassifier** ([mnist-classifier.ts](mnist-classifier.ts:1))

**Main class for the neural network**:
```typescript
class MNISTClassifier {
  private model: tf.LayersModel;

  // Build the network architecture
  buildModel(): void {
    // Define layers
    // Compile with optimizer and loss
  }

  // Train the model
  async train(
    trainData: tf.Tensor,
    trainLabels: tf.Tensor,
    config: TrainingConfig
  ): Promise<TrainingHistory> {
    // Training loop with batching
    // Track loss and accuracy
    // Return history
  }

  // Make predictions
  predict(image: tf.Tensor): number {
    // Forward pass
    // Return predicted digit
  }

  // Evaluate on test set
  evaluate(testData: tf.Tensor, testLabels: tf.Tensor): Metrics {
    // Calculate accuracy, loss
    // Generate confusion matrix
  }

  // Save/load model
  async save(path: string): Promise<void>;
  async load(path: string): Promise<void>;
}
```

#### 2. **DataLoader** ([data-loader.ts](data-loader.ts:1))

**Handles MNIST data loading and preprocessing**:
```typescript
class DataLoader {
  // Load MNIST from TensorFlow.js datasets
  async loadMNIST(): Promise<MNISTData> {
    // Download if needed
    // Split into train/test
    // Return as tensors
  }

  // Normalize pixel values
  normalizeImage(image: tf.Tensor): tf.Tensor {
    // Scale from [0, 255] to [0, 1]
    return image.div(255);
  }

  // Convert labels to one-hot encoding
  oneHotEncode(labels: tf.Tensor, numClasses: number): tf.Tensor {
    // [3] → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    return tf.oneHot(labels, numClasses);
  }

  // Create batches for training
  createBatches(data: tf.Tensor, batchSize: number): tf.Tensor[];
}
```

#### 3. **Utils** ([utils.ts](utils.ts:1))

**Helper functions**:
```typescript
// Calculate metrics
function calculateAccuracy(predictions: number[], labels: number[]): number;
function createConfusionMatrix(predictions: number[], labels: number[]): number[][];

// Visualization
function plotTrainingHistory(history: TrainingHistory): void;
function displayPredictions(images: tf.Tensor, predictions: number[]): void;
function visualizeWeights(model: tf.LayersModel, layerIndex: number): void;
```

---

## Setup and Installation

### Prerequisites

- Node.js v18+
- npm or yarn
- 2GB free disk space (for MNIST data)

### Installation

```bash
# Navigate to project directory
cd phase-2-neural-networks/project-3-mnist

# Install dependencies
npm install

# Build TypeScript
npm run build
```

### Dependencies

```json
{
  "@tensorflow/tfjs-node": "^4.15.0",  // TensorFlow.js with Node.js backend
  "typescript": "^5.3.2",
  "@types/node": "^20.10.0"
}
```

**Why tfjs-node?**
- Much faster than browser version (uses native bindings)
- Can use CPU efficiently
- No browser restrictions

---

## Usage

### Demo 1: Basic Training

**Run**: `npm run demo`

Trains a simple model and makes predictions:
```typescript
// Load data
const data = await dataLoader.loadMNIST();

// Build and train model
const classifier = new MNISTClassifier();
classifier.buildModel();
await classifier.train(data.trainImages, data.trainLabels, {
  epochs: 5,
  batchSize: 128
});

// Test prediction
const testImage = data.testImages.slice([0, 1]);
const prediction = classifier.predict(testImage);
console.log(`Predicted digit: ${prediction}`);
```

**Expected Output**:
```
Loading MNIST dataset...
Building model...
Training...
Epoch 1/5 - loss: 0.4523 - accuracy: 0.8654 - val_loss: 0.1523 - val_accuracy: 0.9541
Epoch 2/5 - loss: 0.1234 - accuracy: 0.9634 - val_loss: 0.0987 - val_accuracy: 0.9698
Epoch 3/5 - loss: 0.0854 - accuracy: 0.9745 - val_loss: 0.0823 - val_accuracy: 0.9743
Epoch 4/5 - loss: 0.0687 - accuracy: 0.9798 - val_loss: 0.0754 - val_accuracy: 0.9765
Epoch 5/5 - loss: 0.0578 - accuracy: 0.9832 - val_loss: 0.0712 - val_accuracy: 0.9781

Training complete! Final accuracy: 97.81%
Predicted digit: 7 (confidence: 98.5%)
```

### Demo 2: Full Training Pipeline

**Run**: `npm run demo:train`

Complete training with visualization:
```typescript
// Train model with callbacks
const history = await classifier.train(trainData, trainLabels, {
  epochs: 10,
  batchSize: 128,
  validationSplit: 0.15,
  callbacks: {
    onEpochEnd: (epoch, logs) => {
      console.log(`Epoch ${epoch}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc.toFixed(4)}`);
    }
  }
});

// Save model
await classifier.save('file://./models/mnist-model');

// Plot results
plotTrainingHistory(history);
```

### Demo 3: Predictions

**Run**: `npm run demo:predict`

Test predictions on multiple images:
```typescript
// Load trained model
await classifier.load('file://./models/mnist-model');

// Get random test images
const numSamples = 10;
const testImages = getRandomSamples(data.testImages, numSamples);
const testLabels = getRandomSamples(data.testLabels, numSamples);

// Make predictions
for (let i = 0; i < numSamples; i++) {
  const image = testImages.slice([i, 1]);
  const prediction = classifier.predict(image);
  const actual = testLabels[i];

  console.log(`Image ${i+1}: Predicted=${prediction}, Actual=${actual}, ${prediction === actual ? '✓' : '✗'}`);
}
```

### Demo 4: Visualization

**Run**: `npm run demo:visualize`

Visualize model internals:
```typescript
// Visualize training progress
plotTrainingHistory(history);

// Show predictions with images
displayPredictions(testImages, predictions);

// Visualize first layer weights
visualizeWeights(model, 0);

// Show confusion matrix
const confusionMatrix = createConfusionMatrix(predictions, labels);
plotConfusionMatrix(confusionMatrix);
```

### Demo 5: Detailed Evaluation

**Run**: `npm run demo:evaluate`

Comprehensive model evaluation:
```typescript
// Evaluate on test set
const metrics = classifier.evaluate(testData, testLabels);

console.log('=== Model Evaluation ===');
console.log(`Test Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
console.log(`Test Loss: ${metrics.loss.toFixed(4)}`);

// Per-class accuracy
console.log('\nPer-Class Accuracy:');
for (let digit = 0; digit < 10; digit++) {
  console.log(`  Digit ${digit}: ${(metrics.classAccuracy[digit] * 100).toFixed(2)}%`);
}

// Confusion matrix
console.log('\nConfusion Matrix:');
console.table(metrics.confusionMatrix);

// Worst predictions
console.log('\nWorst Predictions:');
metrics.worstPredictions.forEach(({ image, predicted, actual, confidence }) => {
  console.log(`  Predicted ${predicted} (${(confidence*100).toFixed(1)}%), Actually ${actual}`);
});
```

---

## Expected Results

### Performance Metrics

**After 10 Epochs**:
- **Training Accuracy**: ~98-99%
- **Test Accuracy**: ~97-98%
- **Training Time**: 3-5 minutes (CPU)
- **Model Size**: ~500 KB

**Confusion Matrix** (typical):
```
     0    1    2    3    4    5    6    7    8    9
0 [ 975   0    1    0    0    1    2    1    0    0 ]
1 [   0 1130   2    1    0    0    1    1    0    0 ]
2 [   2    0 1015   3    2    0    1    5    4    0 ]
3 [   0    0    2  997   0    5    0    3    3    0 ]
4 [   0    0    1    0  970   0    2    1    1    7 ]
5 [   1    0    0   10    0  876   3    0    1    1 ]
6 [   4    2    0    0    2    4  945   0    1    0 ]
7 [   0    2    8    1    1    0    0 1011   2    3 ]
8 [   3    0    3    4    2    2    1    2  956    1 ]
9 [   2    4    0    1   10    3    0    4    2  983 ]

Overall Accuracy: 97.8%
```

**Interpretation**:
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Most confusion between similar digits (3/5, 4/9, 7/2)

### What "Good" Looks Like

**Training Curves**:
```
Loss
  |╲
  | ╲___
  |     ------________
  |__________________ Epochs

Validation loss follows training loss closely → Not overfitting
```

```
Accuracy
  |         _____-----
  |      __/
  |    _/
  |   /
  |____________________ Epochs

Smooth increase, plateaus around 97-98% → Good convergence
```

**Bad Signs**:
- **Training accuracy 99%, Test accuracy 70%**: Overfitting!
- **Both accuracies low (<90%)**: Model too simple or learning rate issues
- **Loss is NaN**: Learning rate too high
- **Loss not decreasing**: Learning rate too low or bug in code

---

## Understanding the Results

### Why 97-98% and Not 100%?

**Reasons**:
1. **Human-level accuracy is ~98%** (some digits are ambiguous even for humans)
2. **Model is relatively simple** (deeper models can reach 99.5%+)
3. **Some images are genuinely difficult**:
   ```
   Is this a messy 5 or a messy 6?
   Is this a 7 with a crossbar or a cursive 2?
   ```

### What Did the Model Learn?

**Layer 1 (128 neurons)**:
- Edge detectors (horizontal, vertical, diagonal)
- Curve detectors
- Basic shapes

**Layer 2 (64 neurons)**:
- Combinations of edges → parts of digits
- Loops (for 0, 6, 8, 9)
- Vertical lines (for 1)
- Horizontal lines with vertical (for 4, 7)

**Output Layer (10 neurons)**:
- Each neuron specialized for one digit
- Combines features from layer 2

**Example Visualization**:
```
Input: Image of "3"
  ↓
Layer 1 detects: top curve, middle curve, bottom curve
  ↓
Layer 2 combines: "two horizontal curves stacked"
  ↓
Output: Neuron 3 activates strongly (97% confidence)
```

---

## Experiments to Try

### 1. **Modify Architecture**

Try different network structures:

**Deeper**:
```typescript
784 → 256 → 128 → 64 → 32 → 10
```
- Expect: Slightly better accuracy, longer training
- Good for: Understanding depth vs performance trade-off

**Wider**:
```typescript
784 → 512 → 512 → 10
```
- Expect: More parameters, may overfit
- Good for: Understanding width vs depth

**Simpler**:
```typescript
784 → 64 → 10
```
- Expect: Lower accuracy (~95%)
- Good for: Minimum viable network

### 2. **Change Hyperparameters**

**Learning Rate**:
```typescript
Try: 0.0001, 0.001, 0.01, 0.1
Observe: Training speed and stability
```

**Batch Size**:
```typescript
Try: 32, 64, 128, 256, 512
Observe: Training time and final accuracy
```

**Epochs**:
```typescript
Try: 5, 10, 20, 50
Observe: When does overfitting start?
```

**Dropout Rate**:
```typescript
Try: 0.0, 0.1, 0.3, 0.5
Observe: Effect on overfitting
```

### 3. **Different Activations**

Replace ReLU with:
- **Sigmoid**: Expect worse performance (vanishing gradients)
- **Tanh**: Expect similar to ReLU
- **Leaky ReLU**: Expect similar or slightly better

### 4. **Regularization**

Add L2 regularization:
```typescript
tf.layers.dense({
  units: 128,
  activation: 'relu',
  kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
})
```

### 5. **Data Augmentation**

Augment training data:
```typescript
// Random rotation ±10 degrees
// Random shift ±2 pixels
// Random zoom 90%-110%

Expect: Better generalization, slower training
```

---

## Common Issues & Solutions

### Issue 1: Low Accuracy (<90%)

**Possible Causes**:
- Forgot to normalize data (images should be [0,1], not [0,255])
- Learning rate too low
- Not enough epochs
- Bug in architecture

**Solutions**:
```typescript
// Check data normalization
const max = trainImages.max().dataSync()[0];
console.log('Max pixel value:', max);  // Should be 1.0, not 255

// Increase epochs
config.epochs = 20;

// Try higher learning rate
config.learningRate = 0.01;
```

### Issue 2: Overfitting (train>>test accuracy)

**Signs**:
```
Training accuracy: 99%
Test accuracy: 94%
Gap > 5% → Overfitting
```

**Solutions**:
```typescript
// Add more dropout
tf.layers.dropout({ rate: 0.5 })

// Add L2 regularization
kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })

// Early stopping
const callbacks = tf.callbacks.earlyStopping({
  monitor: 'val_loss',
  patience: 3
});

// Get more data (or use data augmentation)
```

### Issue 3: Loss is NaN

**Cause**: Learning rate too high, gradients exploded

**Solution**:
```typescript
// Lower learning rate
config.learningRate = 0.0001;

// Use gradient clipping
config.clipNorm = 1.0;
```

### Issue 4: Training is Slow

**Solutions**:
```typescript
// Use tfjs-node-gpu if you have GPU
// Increase batch size
config.batchSize = 256;

// Use fewer neurons
784 → 64 → 10  (instead of 784 → 128 → 64 → 10)
```

### Issue 5: Out of Memory

**Solutions**:
```typescript
// Reduce batch size
config.batchSize = 32;

// Use tf.tidy() to dispose tensors
tf.tidy(() => {
  // Training code here
});

// Dispose tensors explicitly
tensor.dispose();
```

---

## Going Further

### Next Steps

1. **Try CNN**: Convolutional layers preserve spatial structure
2. **Fashion-MNIST**: Same format as MNIST, but clothing items
3. **CIFAR-10**: Color images, harder problem
4. **Custom dataset**: Train on your own handwritten digits

### Advanced Topics

**Convolutional Neural Network**:
```typescript
Input: 28×28×1
  ↓
Conv2D: 32 filters, 3×3, ReLU
  ↓
MaxPooling: 2×2
  ↓
Conv2D: 64 filters, 3×3, ReLU
  ↓
MaxPooling: 2×2
  ↓
Flatten
  ↓
Dense: 128, ReLU
  ↓
Output: 10, Softmax

Expect: 99%+ accuracy
```

**Ensemble Methods**:
- Train 5 models with different initializations
- Average their predictions
- Expect: +0.5-1% accuracy boost

---

## Interview Questions

After completing this project, you should be able to answer:

1. **Explain how your MNIST classifier works**
   - "It's a feedforward neural network with 2 hidden layers using ReLU activation. Input is flattened 28×28 image, output is 10-class softmax. Trained using Adam optimizer with categorical cross-entropy loss."

2. **Why use softmax in the output layer?**
   - "Softmax converts raw scores (logits) into probabilities that sum to 1, perfect for multi-class classification."

3. **What is dropout and why did you use it?**
   - "Dropout randomly deactivates neurons during training, forcing the network to learn redundant representations. This prevents overfitting."

4. **How would you improve the model?**
   - "Use CNNs to preserve spatial structure, add data augmentation, try ensemble methods, tune hyperparameters."

5. **What is the purpose of validation data?**
   - "Monitor for overfitting during training, tune hyperparameters, decide when to stop training."

6. **Explain forward and backward propagation**
   - "Forward: multiply inputs by weights, apply activations, get prediction. Backward: calculate gradients using chain rule, update weights to minimize loss."

7. **Why normalize pixel values?**
   - "Keep inputs in reasonable range for gradient descent. Values in [0,1] prevent exploding/vanishing gradients."

8. **What's the difference between epochs and batches?**
   - "Epoch = one pass through entire dataset. Batch = subset of data for one gradient update. More batches per epoch = more updates."

---

## Resources

### Documentation
- [TensorFlow.js Layers API](https://js.tensorflow.org/api/latest/#Layers)
- [TensorFlow.js Tutorials](https://www.tensorflow.org/js/tutorials)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

### Videos
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [StatQuest - Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw)

### Papers
- [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (1998) - Original CNN for MNIST

---

## Summary

**You built a neural network that**:
- ✅ Recognizes handwritten digits with 97-98% accuracy
- ✅ Uses 2 hidden layers with ReLU activation
- ✅ Prevents overfitting with dropout
- ✅ Trains in minutes on a CPU
- ✅ Can be saved and loaded for reuse

**Key Takeaways**:
1. Neural networks learn hierarchical features automatically
2. Proper data preprocessing (normalization) is critical
3. Dropout and regularization prevent overfitting
4. TensorFlow.js makes deep learning accessible in JavaScript
5. Even simple architectures can achieve excellent results

**Next**: Move to [Project 4: Transfer Learning](../project-4-image-classification/) to build image classifiers with even less data!

---

**Congratulations!** 🎉 You've built your first deep neural network and achieved state-of-the-art results on a classic computer vision task!
