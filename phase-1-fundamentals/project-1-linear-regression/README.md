# Project 1: Linear Regression from Scratch

Build a linear regression model using only TypeScript - no ML libraries!

---

## 📚 **Start Here: Complete Learning Path**

**New to linear regression? Follow this structured learning path:**

### → [**📖 Complete Learning Path Guide**](./LEARNING_PATH.md) ←

This guide provides a step-by-step journey from theory to implementation:
- 📘 Theory documents (what to read and in what order)
- 🔬 Code study approach
- 🧪 Hands-on experiments
- ✅ Learning outcomes checklist
- 🎯 Interview preparation

### Quick Links to Key Documents:

1. **[Linear Regression Theory](./linear-regression-theory.md)** - Complete conceptual guide with real-world examples
2. **[Feature Scaling Explained](./FEATURE_SCALING_EXPLANATION.md)** - Why normalization is critical (fixes NaN errors!)
3. **[Implementation Guide](./README.md)** - This document (project setup and coding)

**Understanding the "why" before the "how" will make implementation much clearer!**

---

## 🎯 Learning Objectives

- Implement gradient descent from scratch
- Understand cost functions (Mean Squared Error)
- Learn matrix operations in JavaScript
- Visualize training progress
- Debug ML algorithms by understanding the math

## 📊 What You'll Build

A complete linear regression system that can:
1. Train on any dataset
2. Make predictions
3. Visualize the regression line
4. Show cost/loss decreasing over iterations
5. Allow tuning hyperparameters (learning rate, iterations)

## 🔢 The Math

### Linear Regression Equation
```
y = mx + b
or more generally:
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

### Cost Function (Mean Squared Error)
```
J(θ) = (1/2m) * Σ(h(xᵢ) - yᵢ)²

where:
- m = number of training examples
- h(x) = predicted value
- y = actual value
```

### Gradient Descent Update Rule
```
θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ

where:
- α = learning rate
- ∂J(θ)/∂θⱼ = partial derivative of cost function
```

For linear regression:
```
∂J(θ)/∂θ₀ = (1/m) * Σ(h(xᵢ) - yᵢ)
∂J(θ)/∂θ₁ = (1/m) * Σ(h(xᵢ) - yᵢ) * xᵢ
```

## 📁 Project Structure

```
project-1-linear-regression/
├── src/
│   ├── linear-regression.ts    # Main model implementation
│   ├── dataset.ts               # Data generation/loading
│   ├── visualize.ts             # Visualization helpers
│   ├── train.ts                 # Training script
│   ├── demo.ts                  # Demo with sample data
│   └── index.ts                 # Entry point
├── data/
│   └── sample-data.csv          # Sample datasets
├── package.json
├── tsconfig.json
└── README.md (this file)
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Demo
```bash
npm run demo
```

### 3. Train on Custom Data
```bash
npm run train
```

## 💡 Implementation Guide

### Step 1: Create the LinearRegression Class

```typescript
// src/linear-regression.ts

export class LinearRegression {
  private theta: number[]; // Model parameters (weights)
  private learningRate: number;
  private iterations: number;
  private costHistory: number[] = [];

  constructor(learningRate = 0.01, iterations = 1000) {
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.theta = [];
  }

  // TODO: Implement these methods
  fit(X: number[][], y: number[]): void {}
  predict(X: number[][]): number[] {}
  computeCost(X: number[][], y: number[]): number {}
  gradientDescent(X: number[][], y: number[]): void {}
}
```

### Step 2: Implement Cost Function

```typescript
computeCost(X: number[][], y: number[]): number {
  const m = y.length;
  const predictions = this.predict(X);

  // TODO: Calculate Mean Squared Error
  // MSE = (1/2m) * Σ(prediction - actual)²

  return cost;
}
```

### Step 3: Implement Gradient Descent

```typescript
gradientDescent(X: number[][], y: number[]): void {
  const m = y.length;

  for (let iter = 0; iter < this.iterations; iter++) {
    // TODO:
    // 1. Calculate predictions
    // 2. Calculate errors
    // 3. Calculate gradients
    // 4. Update theta (weights)
    // 5. Calculate and store cost

    // Log progress every 100 iterations
    if (iter % 100 === 0) {
      console.log(`Iteration ${iter}, Cost: ${this.computeCost(X, y)}`);
    }
  }
}
```

### Step 4: Implement Prediction

```typescript
predict(X: number[][]): number[] {
  // TODO: Calculate h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ...
  // Use matrix multiplication if using mathjs

  return predictions;
}
```

## 📈 Datasets to Try

### 1. Simple Linear (y = 2x + 3)
```typescript
const X = [[1], [2], [3], [4], [5]];
const y = [5, 7, 9, 11, 13];
```

### 2. House Prices
```typescript
// Size (sqft) → Price ($1000s)
const X = [[1000], [1500], [2000], [2500], [3000]];
const y = [200, 300, 400, 500, 600];
```

### 3. Multivariate (Multiple Features)
```typescript
// [size, bedrooms] → price
const X = [
  [1000, 2],
  [1500, 3],
  [2000, 3],
  [2500, 4],
  [3000, 4]
];
const y = [200, 300, 400, 500, 600];
```

## 🎨 Visualization

Use Chart.js to visualize:
1. **Training data vs predictions** (scatter plot + line)
2. **Cost over iterations** (line plot showing convergence)

```typescript
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';

// TODO: Implement visualization
```

## ✅ Success Criteria

Your implementation is complete when:

- [ ] Model can train on simple linear data
- [ ] Cost decreases with each iteration
- [ ] Predictions are accurate (low error)
- [ ] Can handle multiple features (multivariate regression)
- [ ] Visualization shows:
  - [ ] Scatter plot of data
  - [ ] Regression line
  - [ ] Cost history (decreasing curve)
- [ ] Can tune learning rate and see impact
- [ ] Code is well-documented with comments

## 🐛 Debugging Tips

### Problem: Cost is increasing
**Solution**: Learning rate too high. Try smaller values (0.001, 0.0001)

### Problem: Cost not changing
**Solution**: Learning rate too low. Try larger values (0.1, 0.5)

### Problem: NaN errors
**Solution**:
- Check for division by zero
- Normalize features (scale to similar ranges)
- Verify gradient calculations

### Problem: Slow convergence
**Solution**:
- Feature scaling (normalize inputs)
- Increase iterations
- Adjust learning rate

## 🧪 Testing Your Implementation

### Test 1: Perfect Linear Data
```typescript
// Should get perfect fit (cost ≈ 0)
const X = [[1], [2], [3], [4], [5]];
const y = [2, 4, 6, 8, 10]; // Perfect y = 2x

// Expected: theta ≈ [0, 2]
```

### Test 2: With Noise
```typescript
// Should still fit reasonably well
const X = [[1], [2], [3], [4], [5]];
const y = [2.1, 3.9, 6.2, 7.8, 10.1]; // y = 2x + noise

// Expected: theta ≈ [0, 2]
```

## 📚 Additional Challenges

Once you have basic implementation working:

1. **Feature Normalization**: Implement mean normalization
2. **Polynomial Regression**: Add polynomial features (x², x³)
3. **Regularization**: Add L2 regularization to prevent overfitting
4. **Analytical Solution**: Implement normal equation (no iteration needed!)
5. **Batch vs Stochastic**: Try mini-batch gradient descent
6. **Web Interface**: Create HTML/React frontend to visualize live

## 🎯 Interview Questions

After completing this project, you should be able to answer:

1. **Walk me through how gradient descent works**
2. **Why do we need a cost function?**
3. **What happens if learning rate is too high? Too low?**
4. **How do you know if your model is converging?**
5. **Explain the difference between batch and stochastic gradient descent**
6. **Why is feature scaling important?**
7. **When would you use normal equation vs gradient descent?**

## 📖 Resources

### 📘 Project Documentation (Read These First!)
- **[Linear Regression Theory Guide](./linear-regression-theory.md)** ⭐ - Complete conceptual explanation with examples
- **[Feature Scaling Explained](./FEATURE_SCALING_EXPLANATION.md)** ⭐ - Why normalization is critical

### 🎥 Video Resources
- [Andrew Ng's ML Course - Linear Regression](https://www.coursera.org/learn/machine-learning)
- [3Blue1Brown - Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [StatQuest - Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)

## 🚀 Next Steps

After completing this project:
1. Deploy to GitHub Pages or Vercel
2. Write a blog post explaining gradient descent
3. Move on to Project 2: K-Means Clustering
4. Update your portfolio with this project

## 💡 Tips

- **Start simple**: Get basic version working first, then add features
- **Visualize early**: Seeing data helps debug
- **Calculate by hand**: Do one gradient descent step manually to verify code
- **Log everything**: Print theta and cost at each step initially
- **Compare with library**: Use simple-statistics or ml.js to verify your results

---

**Ready to code?** Start with `src/linear-regression.ts` and implement the LinearRegression class!
