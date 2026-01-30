# Phase 1: AI Fundamentals & Math Essentials
## Weeks 1-2: Build Strong Foundation

**Goal**: Understand the "why" behind AI, not just the "how"

---

## Overview

This phase focuses on building a solid mathematical and conceptual foundation. You'll implement ML algorithms from scratch without using ML libraries to truly understand how they work.

**Time Commitment**: 15-25 hours over 2 weeks

---

## Learning Objectives

By the end of Phase 1, you will be able to:
- ✅ Explain the differences between AI, ML, and Deep Learning
- ✅ Understand gradient descent and backpropagation
- ✅ Implement linear regression from scratch
- ✅ Implement K-means clustering from scratch
- ✅ Perform matrix operations using JavaScript
- ✅ Visualize data and model predictions
- ✅ Debug ML algorithms by understanding the math

---

## Week 1: Linear Algebra, Calculus & Linear Regression

### Theory Topics

#### 1. What is AI, ML, Deep Learning?
- **AI (Artificial Intelligence)**: Systems that can perform tasks requiring human intelligence
- **ML (Machine Learning)**: Subset of AI - systems that learn from data
- **Deep Learning**: Subset of ML - neural networks with multiple layers

**Key Distinctions**:
```
AI (Broadest)
 └─ Machine Learning
     └─ Deep Learning (Neural Networks)
```

**Learning Types**:
- **Supervised Learning**: Learn from labeled data (most common)
- **Unsupervised Learning**: Find patterns in unlabeled data
- **Reinforcement Learning**: Learn through trial and error with rewards

#### 2. Linear Algebra Basics
**Essential Concepts**:
- Vectors and vector operations
- Matrices and matrix multiplication
- Dot product and its geometric meaning
- Matrix transpose
- Identity matrix

**Why It Matters**:
- Data is represented as vectors/matrices
- Model weights are matrices
- Forward pass = matrix multiplication
- Understanding this helps debug ML code

**Resources**:
- 📺 [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (Watch episodes 1-3)
- 🔧 Practice with Math.js library

#### 3. Calculus Fundamentals
**Essential Concepts**:
- Derivatives (rate of change)
- Gradients (derivative in multiple dimensions)
- Chain rule (backpropagation relies on this!)
- Partial derivatives

**Why It Matters**:
- Optimization uses gradients
- Training = minimizing error using calculus
- Backpropagation is just chain rule applied

**Resources**:
- 📺 [3Blue1Brown - Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (Watch episodes 1-4)

#### 4. Gradient Descent
**The Core of ML Training**:
```
1. Start with random weights
2. Calculate error (loss)
3. Calculate gradient (direction to reduce error)
4. Update weights: weight = weight - learning_rate * gradient
5. Repeat until error is small enough
```

**Hyperparameters**:
- Learning rate (how big steps to take)
- Number of iterations/epochs

### Project 1: Linear Regression from Scratch

**Objective**: Build a linear regression model using only TypeScript (no ML libraries)

**What You'll Build**:
- Gradient descent implementation
- Cost function (Mean Squared Error)
- Prediction function
- Visualization of data and regression line

**Dataset Ideas**:
- House price prediction (size → price)
- Temperature conversion (Celsius → Fahrenheit)
- Student hours vs exam score

**Skills Gained**:
- Matrix operations in JavaScript
- Implementing gradient descent
- Understanding learning rates
- Data visualization with Chart.js/D3.js

**Location**: `./project-1-linear-regression/`

**Starter Code**: See project folder

---

## Week 2: Probability, Statistics & Clustering

### Theory Topics

#### 1. Probability and Statistics
**Essential Concepts**:
- Mean, median, mode, variance, standard deviation
- Normal distribution
- Probability basics
- Conditional probability

**Why It Matters**:
- Understand data distributions
- Model uncertainty
- Evaluate model performance
- Anomaly detection

**Resources**:
- 📺 [StatQuest - Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)

#### 2. Cost Functions and Optimization
**Cost Function (Loss Function)**:
- Measures how wrong your model is
- Goal: minimize this function
- Different problems use different cost functions

**Common Cost Functions**:
- **MSE (Mean Squared Error)**: For regression
- **Cross-Entropy**: For classification
- **MAE (Mean Absolute Error)**: For regression (robust to outliers)

#### 3. Supervised vs Unsupervised Learning
**Supervised Learning**:
- Training data has labels (answers)
- Examples: Classification, regression
- Goal: Learn mapping from input → output

**Unsupervised Learning**:
- No labels in training data
- Examples: Clustering, dimensionality reduction
- Goal: Find patterns and structure

### Project 2: K-Means Clustering

**Objective**: Implement K-means clustering algorithm from scratch

**What You'll Build**:
- K-means algorithm implementation
- Distance calculation (Euclidean distance)
- Cluster visualization
- Apply to real dataset

**Algorithm Steps**:
```
1. Initialize K random centroids
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat steps 2-3 until convergence
```

**Dataset Ideas**:
- Customer segmentation (2D data for easy visualization)
- Iris dataset (classic ML dataset)
- Color clustering from images

**Skills Gained**:
- Implementing unsupervised learning
- Working with unlabeled data
- Visualization with Canvas/D3.js
- Understanding convergence

**Location**: `./project-2-kmeans-clustering/`

**Starter Code**: See project folder

---

## Projects Overview

| Project | Type | Difficulty | Time | Key Concepts |
|---------|------|------------|------|--------------|
| **Project 1: Linear Regression** | Supervised | ⭐⭐ | 8-12 hours | Gradient descent, cost functions |
| **Project 2: K-Means Clustering** | Unsupervised | ⭐⭐ | 6-10 hours | Clustering, distance metrics |

---

## Tools & Technologies

### Required
- **Node.js** v18+ (JavaScript runtime)
- **TypeScript** (type safety)
- **Math.js** (matrix operations)
- **Chart.js** or **D3.js** (data visualization)

### Setup Instructions

```bash
# Navigate to Phase 1
cd phase-1-fundamentals

# Install dependencies for Project 1
cd project-1-linear-regression
npm install

# Install dependencies for Project 2
cd ../project-2-kmeans-clustering
npm install
```

---

## Resources

### Video Courses (FREE)
1. **3Blue1Brown - Essence of Linear Algebra** ⭐⭐⭐
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
   - Watch: Episodes 1-3 (vectors, linear combinations, matrix multiplication)

2. **3Blue1Brown - Essence of Calculus** ⭐⭐⭐
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
   - Watch: Episodes 1-4 (derivatives, chain rule)

3. **StatQuest - Statistics Fundamentals** ⭐⭐
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
   - Watch: Mean/variance, Normal distribution

### Documentation
- [Math.js Documentation](https://mathjs.org/docs/)
- [Chart.js Documentation](https://www.chartjs.org/docs/)
- [D3.js Documentation](https://d3js.org/)

### Interactive Learning
- [Seeing Theory - Visual Probability](https://seeing-theory.brown.edu/)
- [Matrix Multiplication Visualizer](http://matrixmultiplication.xyz/)

---

## Weekly Checklist

### Week 1
- [ ] Watch 3Blue1Brown Linear Algebra (Episodes 1-3)
- [ ] Watch 3Blue1Brown Calculus (Episodes 1-4)
- [ ] Read: What is AI, ML, Deep Learning
- [ ] Complete Project 1: Linear Regression
- [ ] Deploy Project 1 (GitHub Pages or Vercel)
- [ ] Write reflection in learning journal

### Week 2
- [ ] Watch StatQuest Statistics videos
- [ ] Read about supervised vs unsupervised learning
- [ ] Complete Project 2: K-Means Clustering
- [ ] Deploy Project 2
- [ ] Write reflection in learning journal
- [ ] Update PROGRESS.md

---

## Success Criteria

You've successfully completed Phase 1 when you can:

### Technical Understanding
- ✅ Explain gradient descent to someone without ML background
- ✅ Calculate gradients by hand for simple functions
- ✅ Implement linear regression without looking at code
- ✅ Debug why a model isn't converging (learning rate, iterations)

### Practical Skills
- ✅ Both projects working and deployed
- ✅ Can visualize data effectively
- ✅ Comfortable with matrix operations in JavaScript
- ✅ Can read and understand ML pseudocode

### Interview Readiness
- ✅ Can whiteboard linear regression algorithm
- ✅ Can explain the math behind gradient descent
- ✅ Can discuss supervised vs unsupervised learning
- ✅ Can explain when to use different cost functions

---

## Common Pitfalls & Solutions

### Problem: Gradient descent not converging
**Solutions**:
- Lower learning rate (try 0.01, 0.001)
- Normalize your data (scale features to similar ranges)
- Check for bugs in gradient calculation
- Increase number of iterations

### Problem: Understanding the math
**Solutions**:
- Focus on intuition first, rigor later
- Draw diagrams of concepts
- Implement in code (makes math concrete)
- Watch multiple explanations (everyone explains differently)

### Problem: Visualization not working
**Solutions**:
- Start with simple 2D plots
- Use Chart.js for easier setup
- D3.js has steeper learning curve but more power
- Check browser console for errors

---

## Interview Questions to Practice

After Phase 1, you should be able to answer:

1. **What is the difference between AI, ML, and Deep Learning?**
2. **Explain gradient descent like I'm 5 years old.**
3. **What is a cost function and why do we need it?**
4. **How does linear regression work mathematically?**
5. **What's the difference between supervised and unsupervised learning?**
6. **When would you use K-means clustering?**
7. **What are the hyperparameters in gradient descent?**
8. **How do you know if your model is converging?**

---

## Next Steps

After completing Phase 1:
1. ✅ Update [PROGRESS.md](../PROGRESS.md)
2. ✅ Write a blog post about your learnings
3. ✅ Share your projects on LinkedIn/Twitter
4. ✅ Move to [Phase 2: Neural Networks & TensorFlow.js](../phase-2-neural-networks/)

---

## Tips for Success

1. **Don't rush the math** - It pays off in later phases
2. **Implement before using libraries** - Understanding > convenience
3. **Visualize everything** - Seeing data helps intuition
4. **Ask "why" repeatedly** - Don't just memorize formulas
5. **Debug by hand** - Calculate gradients manually to verify code

**Remember**: This phase is about building intuition. You don't need to be a mathematician, but you should understand *why* algorithms work, not just *that* they work.

---

**Ready?** Start with Project 1: Linear Regression!

[→ Project 1: Linear Regression](./project-1-linear-regression/)
