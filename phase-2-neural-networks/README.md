# Phase 2: Neural Networks & TensorFlow.js
## Weeks 3-4: Deep Learning Fundamentals

**Goal**: Understand how neural networks work and build practical deep learning applications

---

## Overview

This phase transitions from traditional ML to **deep learning**. You'll learn how neural networks work under the hood, then build three practical applications using TensorFlow.js.

**Time Commitment**: 20-30 hours over 2 weeks

---

## Learning Objectives

By the end of Phase 2, you will be able to:
- ✅ Explain how neural networks learn (forward & backward propagation)
- ✅ Understand activation functions and why they matter
- ✅ Build and train neural networks with TensorFlow.js
- ✅ Recognize handwritten digits (image classification)
- ✅ Use transfer learning with pre-trained models
- ✅ Perform sentiment analysis on text data
- ✅ Debug neural network training issues
- ✅ Visualize training progress and model performance

---

## Week 3: Neural Network Fundamentals & MNIST

### Theory Topics

#### 1. What are Neural Networks?

**Simple Definition**:
> Neural networks are algorithms inspired by the brain that learn to recognize patterns through examples.

**Structure**:
```
Input Layer → Hidden Layers → Output Layer
```

**How They Differ from Linear Regression**:
- **Linear Regression**: Single line/plane, only linear patterns
- **Neural Networks**: Multiple layers, can learn complex non-linear patterns

#### 2. Neural Network Architecture

**Components**:
- **Neurons**: Basic computation units
- **Weights**: Connection strengths between neurons
- **Biases**: Shift in activation
- **Activation Functions**: Add non-linearity (sigmoid, ReLU, tanh)

**Visual Example**:
```
Input      Hidden      Output
Layer      Layer       Layer

x₁ ──┐    ┌─── h₁ ───┐
     ├────┤           ├──→ ŷ
x₂ ──┘    └─── h₂ ───┘
```

Each connection has a weight, each neuron has a bias.

#### 3. Activation Functions

**Why We Need Them**:
Without activation functions, neural networks are just fancy linear regression!

**Common Activation Functions**:

1. **ReLU (Rectified Linear Unit)** - Most popular
   ```
   f(x) = max(0, x)
   ```
   - Fast to compute
   - Fixes "vanishing gradient" problem
   - Default choice for hidden layers

2. **Sigmoid** - Classic but less used now
   ```
   f(x) = 1 / (1 + e⁻ˣ)
   ```
   - Output between 0 and 1
   - Good for binary classification output layer
   - Suffers from vanishing gradients

3. **Softmax** - For multi-class classification
   ```
   f(xᵢ) = e^xᵢ / Σe^xⱼ
   ```
   - Converts logits to probabilities
   - Outputs sum to 1
   - Used in output layer for classification

4. **Tanh** - Centered around zero
   ```
   f(x) = (e^x - e⁻ˣ) / (e^x + e⁻ˣ)
   ```
   - Output between -1 and 1
   - Better than sigmoid for hidden layers

#### 4. Forward Propagation

**The Process**:
1. Start with input data
2. Multiply by weights, add biases
3. Apply activation function
4. Pass to next layer
5. Repeat until output layer
6. Get prediction

**Example (2-layer network)**:
```
Input: x = [x₁, x₂]

Hidden layer:
h = ReLU(W₁ · x + b₁)

Output layer:
ŷ = softmax(W₂ · h + b₂)
```

#### 5. Backward Propagation (Backprop)

**The Core of Neural Network Training**:
> Backpropagation calculates how much each weight contributed to the error, then updates weights to reduce that error.

**Process**:
1. Calculate error at output layer
2. Propagate error backward through network
3. Calculate gradient for each weight
4. Update weights using gradient descent

**Why It's Called "Backpropagation"**:
- Forward pass: Input → Output (make prediction)
- Backward pass: Output → Input (calculate gradients)

**Mathematical Foundation**:
- Uses chain rule from calculus
- Same gradient descent as linear regression
- But with multiple layers!

#### 6. Loss Functions for Neural Networks

**For Classification**:
- **Categorical Cross-Entropy**: Multi-class (MNIST - 10 digits)
  ```
  Loss = -Σ yᵢ · log(ŷᵢ)
  ```
- **Binary Cross-Entropy**: Binary classification (sentiment: positive/negative)
  ```
  Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  ```

**For Regression**:
- **MSE (Mean Squared Error)**: Same as linear regression

**Resources**:
- 📺 [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (Watch all 4 episodes - **ESSENTIAL**)
- 📺 [StatQuest - Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw)

---

### Project 3: MNIST Handwritten Digit Recognition

**Objective**: Build a neural network that recognizes handwritten digits (0-9)

**What You'll Build**:
- Feedforward neural network with TensorFlow.js
- Train on 60,000 digit images
- Achieve >95% accuracy
- Real-time prediction interface
- Visualization of learned features

**Dataset**: MNIST
- 60,000 training images
- 10,000 test images
- 28×28 grayscale images
- 10 classes (digits 0-9)

**Architecture**:
```
Input: 784 neurons (28×28 flattened)
   ↓
Hidden Layer 1: 128 neurons, ReLU
   ↓
Hidden Layer 2: 64 neurons, ReLU
   ↓
Output: 10 neurons, Softmax (one per digit)
```

**What You'll Learn**:
- Loading and preprocessing image data
- Building neural networks with TensorFlow.js
- Training with batching and epochs
- Evaluating accuracy and loss
- Visualizing predictions and errors
- Overfitting vs underfitting

**Success Criteria**:
- Train accuracy > 98%
- Test accuracy > 95%
- Can predict custom drawn digits

**Location**: `./project-3-mnist/`

---

## Week 4: Transfer Learning & NLP

### Theory Topics

#### 1. Convolutional Neural Networks (CNNs)

**What They Are**:
> CNNs are neural networks designed for image processing that learn spatial features automatically.

**Key Concepts**:
- **Convolutional Layers**: Detect features (edges, shapes, objects)
- **Pooling Layers**: Reduce size, keep important info
- **Filters/Kernels**: Slide across image to detect patterns

**Why They're Powerful**:
- Automatically learn features (no manual feature engineering!)
- Translation invariant (detect cat anywhere in image)
- Hierarchical features (edges → shapes → objects)

**Architecture Pattern**:
```
Input Image
   ↓
Conv → ReLU → Pool
   ↓
Conv → ReLU → Pool
   ↓
Flatten
   ↓
Dense → ReLU
   ↓
Output (Softmax)
```

#### 2. Transfer Learning

**The Idea**:
> Use a model trained on millions of images and adapt it for your specific task.

**Why It's Powerful**:
- Train with little data (hundreds instead of millions)
- Much faster training
- Often better accuracy
- Don't need expensive GPUs

**How It Works**:
1. Take pre-trained model (e.g., MobileNet trained on ImageNet)
2. Remove last layer (classification head)
3. Add your own classification layer
4. Train only your layer (freeze pre-trained weights)
5. Optionally: fine-tune entire model

**Popular Pre-trained Models**:
- **MobileNet**: Lightweight, fast (good for web/mobile)
- **ResNet**: Very accurate (larger)
- **EfficientNet**: Best accuracy/size trade-off
- **Inception**: Google's architecture

**When to Use**:
- Small dataset (< 10,000 images)
- Limited compute resources
- Need fast results
- Domain similar to ImageNet (photos, objects)

#### 3. Natural Language Processing (NLP) Basics

**What is NLP**:
> Teaching computers to understand and process human language.

**Key Concepts**:
- **Tokenization**: Split text into words/subwords
- **Embeddings**: Convert words to numbers (vectors)
- **Sequence Models**: Process text in order (RNNs, LSTMs)

**Word Embeddings**:
```
"cat" → [0.2, 0.9, -0.1, ...]  (dense vector)
"dog" → [0.3, 0.8, -0.2, ...]  (similar to cat)
"car" → [-0.5, -0.3, 0.7, ...] (different from cat)
```

**Similar words have similar vectors!**

#### 4. Recurrent Neural Networks (RNNs)

**What They Are**:
> Neural networks designed for sequences (text, time series).

**Key Feature**:
- **Memory**: Each step sees previous steps
- **Sequential Processing**: Process one word at a time
- **Context**: "I love this movie" vs "I hate this movie"

**Architecture**:
```
Input: "This movie is great"

t₁: "This"   → h₁ (hidden state)
t₂: "movie"  → h₂ (uses h₁)
t₃: "is"     → h₃ (uses h₂)
t₄: "great"  → h₄ (uses h₃)
                ↓
            Sentiment: Positive
```

**Resources**:
- 📺 [Stanford CS231n - CNNs](http://cs231n.stanford.edu/) (Lectures 5-9)
- 📺 [Stanford CS224n - NLP](http://web.stanford.edu/class/cs224n/) (Lectures 1-3)

---

### Project 4: Image Classification with Transfer Learning

**Objective**: Build a custom image classifier using transfer learning

**What You'll Build**:
- Use MobileNet (pre-trained on ImageNet)
- Classify custom categories (e.g., cats vs dogs)
- Web interface for uploading images
- Train with small dataset (~100 images per class)
- Achieve >90% accuracy

**Architecture**:
```
MobileNet (frozen)
   ↓
Global Average Pooling
   ↓
Dense Layer (128 neurons, ReLU)
   ↓
Dropout (0.5)
   ↓
Output (num_classes, Softmax)
```

**Dataset Options**:
1. **Cats vs Dogs** (binary classification)
2. **Food-101** (food categories)
3. **Your own dataset** (collect your own images!)

**What You'll Learn**:
- Loading pre-trained models in TensorFlow.js
- Freezing/unfreezing layers
- Data augmentation (flip, rotate, zoom)
- Handling real-world messy data
- Overfitting prevention (dropout, early stopping)

**Success Criteria**:
- Test accuracy > 90%
- Works on new images not in training set
- Can explain why transfer learning worked

**Location**: `./project-4-image-classification/`

---

### Project 5: Sentiment Analysis

**Objective**: Build a model that determines if text is positive or negative

**What You'll Build**:
- RNN/LSTM for text classification
- Predict sentiment of movie reviews
- Web interface for testing custom text
- Achieve >85% accuracy
- Visualize word importance

**Dataset**: IMDB Movie Reviews
- 25,000 training reviews
- 25,000 test reviews
- Binary classification (positive/negative)

**Architecture**:
```
Input: Text (e.g., "This movie was amazing!")
   ↓
Tokenization → [23, 145, 67, 892, 12]
   ↓
Embedding Layer (learn word vectors)
   ↓
LSTM Layer (128 units)
   ↓
Dense Layer (64 neurons, ReLU)
   ↓
Output (1 neuron, Sigmoid)
   ↓
Prediction: 0.92 (Positive!)
```

**What You'll Learn**:
- Text preprocessing and tokenization
- Word embeddings (word2vec concept)
- Building and training LSTMs
- Handling variable-length sequences
- Interpreting model predictions

**Success Criteria**:
- Test accuracy > 85%
- Works on custom text you write
- Can identify key positive/negative words

**Location**: `./project-5-sentiment-analysis/`

---

## Projects Overview

| Project | Type | Difficulty | Time | Key Concepts |
|---------|------|------------|------|--------------|
| **Project 3: MNIST** | Computer Vision | ⭐⭐ | 8-12 hours | Neural networks, image classification |
| **Project 4: Transfer Learning** | Computer Vision | ⭐⭐⭐ | 8-12 hours | CNNs, transfer learning, data augmentation |
| **Project 5: Sentiment Analysis** | NLP | ⭐⭐⭐ | 10-14 hours | RNNs/LSTMs, embeddings, text processing |

---

## Tools & Technologies

### Required
- **TensorFlow.js** - Deep learning in JavaScript
- **Node.js** v18+
- **TypeScript** - Type safety
- **tfjs-node** or **tfjs-node-gpu** - Backend for training

### Optional but Recommended
- **@tensorflow-models/mobilenet** - Pre-trained models
- **@tensorflow-models/universal-sentence-encoder** - Text embeddings
- **canvas** - For image manipulation
- **sharp** - Image preprocessing

### Setup Instructions

```bash
# Navigate to Phase 2
cd phase-2-neural-networks

# Install dependencies for Project 3
cd project-3-mnist
npm install

# Install dependencies for Project 4
cd ../project-4-image-classification
npm install

# Install dependencies for Project 5
cd ../project-5-sentiment-analysis
npm install
```

---

## Resources

### Video Courses (FREE)

1. **3Blue1Brown - Neural Networks** ⭐⭐⭐ **MUST WATCH**
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
   - Watch: ALL 4 episodes (absolutely essential!)
   - Best visual explanation of how neural networks work

2. **StatQuest - Neural Networks** ⭐⭐
   - [Neural Networks Playlist](https://www.youtube.com/watch?v=CqOfi41LfDw)
   - Great supplementary explanations

3. **TensorFlow.js Official Tutorials** ⭐⭐⭐
   - [Get Started](https://www.tensorflow.org/js/tutorials)
   - Official examples and guides

4. **Stanford CS231n - CNNs** (Advanced)
   - [Course Website](http://cs231n.stanford.edu/)
   - Lectures 5-9 for deep dive into CNNs

5. **Fast.ai - Practical Deep Learning** ⭐⭐⭐
   - [Course Link](https://course.fast.ai/)
   - Top-down learning approach

### Documentation
- [TensorFlow.js API](https://js.tensorflow.org/api/latest/)
- [TensorFlow.js Models](https://github.com/tensorflow/tfjs-models)
- [Keras Guide](https://www.tensorflow.org/guide/keras) (concepts apply to TF.js)

### Interactive Learning
- [TensorFlow Playground](https://playground.tensorflow.org/) - Visualize neural networks
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Interactive CNN visualization
- [Neural Network Playground](https://playground.tensorflow.org/)

### Books (Optional)
- **Deep Learning with JavaScript** by Shanqing Cai et al.
- **Hands-On Machine Learning** by Aurélien Géron (Python but concepts apply)

---

## Weekly Checklist

### Week 3
- [ ] Watch 3Blue1Brown Neural Networks series (ALL 4 episodes)
- [ ] Read neural network theory (see `neural-network-theory.md`)
- [ ] Complete Project 3: MNIST
- [ ] Achieve >95% test accuracy on MNIST
- [ ] Understand forward and backward propagation
- [ ] Write reflection in learning journal

### Week 4
- [ ] Watch Stanford CS231n lectures on CNNs (optional but recommended)
- [ ] Read about transfer learning and CNNs
- [ ] Complete Project 4: Image Classification
- [ ] Complete Project 5: Sentiment Analysis
- [ ] Deploy at least one project
- [ ] Update PROGRESS.md
- [ ] Write blog post about neural networks

---

## Success Criteria

You've successfully completed Phase 2 when you can:

### Technical Understanding
- ✅ Explain backpropagation to someone without ML background
- ✅ Describe why activation functions are necessary
- ✅ Know when to use different architectures (dense, CNN, RNN)
- ✅ Debug neural network training issues (learning rate, overfitting, etc.)

### Practical Skills
- ✅ All three projects working with good accuracy
- ✅ Comfortable with TensorFlow.js API
- ✅ Can use pre-trained models effectively
- ✅ Can preprocess images and text for neural networks
- ✅ Understand training curves and what they mean

### Interview Readiness
- ✅ Can explain how neural networks learn
- ✅ Can describe the difference between CNN, RNN, and feedforward networks
- ✅ Can explain transfer learning and when to use it
- ✅ Can discuss overfitting prevention techniques
- ✅ Can walk through backpropagation algorithm

---

## Common Pitfalls & Solutions

### Problem: Model not learning (loss not decreasing)
**Solutions**:
- Check learning rate (try 0.001, 0.01, 0.1)
- Verify data preprocessing (normalization, scaling)
- Check for bugs in loss function or architecture
- Try simpler model first (fewer layers)
- Ensure labels are correct

### Problem: Overfitting (train accuracy ≫ test accuracy)
**Solutions**:
- Add dropout layers (0.3-0.5)
- Use data augmentation
- Get more training data
- Reduce model complexity (fewer neurons/layers)
- Use early stopping
- Add L2 regularization

### Problem: Underfitting (both accuracies low)
**Solutions**:
- Increase model complexity (more layers/neurons)
- Train for more epochs
- Reduce regularization (dropout, L2)
- Check if data has enough signal
- Try different architecture

### Problem: Training is very slow
**Solutions**:
- Use tfjs-node instead of browser
- Use tfjs-node-gpu if you have GPU
- Reduce batch size if out of memory
- Use smaller model
- Try transfer learning instead of training from scratch

### Problem: Out of memory errors
**Solutions**:
- Reduce batch size
- Use smaller model
- Process data in chunks
- Use tf.tidy() to clean up tensors
- Dispose tensors explicitly

---

## Interview Questions to Practice

After Phase 2, you should be able to answer:

1. **Explain how a neural network learns**
2. **What is backpropagation and why is it important?**
3. **Why do we need activation functions?**
4. **What's the difference between ReLU and Sigmoid?**
5. **When would you use a CNN vs an RNN?**
6. **Explain transfer learning and when to use it**
7. **What causes overfitting and how do you prevent it?**
8. **What's the difference between batch gradient descent and stochastic gradient descent?**
9. **How do you choose the number of layers and neurons?**
10. **What is dropout and why does it work?**

---

## Performance Benchmarks

### Project 3: MNIST
- **Expected Accuracy**: 95-98%
- **Training Time**: 5-10 minutes (CPU)
- **Model Size**: ~500 KB

### Project 4: Transfer Learning
- **Expected Accuracy**: 90-95%
- **Training Time**: 2-5 minutes
- **Model Size**: ~10 MB (with MobileNet)

### Project 5: Sentiment Analysis
- **Expected Accuracy**: 85-90%
- **Training Time**: 10-20 minutes
- **Model Size**: ~5 MB

---

## Next Steps

After completing Phase 2:
1. ✅ Update [PROGRESS.md](../PROGRESS.md)
2. ✅ Write a blog post comparing Phase 1 (traditional ML) vs Phase 2 (deep learning)
3. ✅ Share your projects on LinkedIn/Twitter
4. ✅ Deploy at least one project (Vercel, Netlify, GitHub Pages)
5. ✅ Move to [Phase 3: AI Domain Overview](../phase-3-domain-overview/)

---

## Tips for Success

1. **Watch 3Blue1Brown first** - Best investment of 1 hour
2. **Start simple** - Get basic neural network working before adding complexity
3. **Visualize everything** - Plot training curves, look at predictions
4. **Understand, don't memorize** - Know why architectures work
5. **Experiment** - Try different hyperparameters, architectures
6. **Debug systematically** - Check data → model → training loop
7. **Use transfer learning** - Don't train from scratch if you don't have to
8. **Monitor training** - Watch loss and accuracy curves in real-time

**Remember**: Deep learning is more empirical than traditional ML. You'll need to experiment and iterate to find what works best.

---

## Comparison: Phase 1 vs Phase 2

| Aspect | Phase 1 (Traditional ML) | Phase 2 (Deep Learning) |
|--------|-------------------------|------------------------|
| **Algorithms** | Linear Regression, K-Means | Neural Networks, CNNs, RNNs |
| **Features** | Manual engineering | Learned automatically |
| **Data Needs** | Can work with small data | Needs more data (or transfer learning) |
| **Interpretability** | Easy to interpret | Harder (black box) |
| **Power** | Linear patterns | Complex non-linear patterns |
| **Training Time** | Fast (seconds) | Slower (minutes to hours) |
| **When to Use** | Simple problems, need interpretability | Complex patterns, lots of data |

**Key Insight**: Phase 1 algorithms are **interpretable and fast**, Phase 2 algorithms are **powerful and flexible**. Use the simplest tool that solves your problem!

---

**Ready?** Start with Project 3: MNIST Digit Recognition!

[→ Project 3: MNIST](./project-3-mnist/)
