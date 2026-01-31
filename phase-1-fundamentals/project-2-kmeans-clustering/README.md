# Project 2: K-Means Clustering from Scratch

Build a K-Means clustering algorithm using only TypeScript - your first unsupervised learning project!

---

## 📚 **Start Here: Complete Learning Path**

**New to K-Means clustering? Follow this structured approach:**

### → [**📖 K-Means Theory Guide**](./k-means-theory.md) ←

**Read this first!** Comprehensive guide covering:
- ✅ What is clustering? (Supervised vs Unsupervised)
- ✅ How K-Means works (step-by-step with visuals)
- ✅ Real-world examples (customer segmentation, image compression)
- ✅ Distance metrics (Euclidean explained)
- ✅ Choosing K (elbow method, silhouette score)
- ✅ Initialization strategies (K-Means++)
- ✅ Convergence criteria
- ✅ Advantages, limitations, and common pitfalls

**Understanding unsupervised learning is KEY before coding!**

---

## 🎯 Learning Objectives

- Implement K-Means clustering algorithm from scratch
- Understand unsupervised learning (no labels!)
- Learn distance calculations (Euclidean distance)
- Master centroid initialization (K-Means++)
- Visualize clusters in 2D
- Choose optimal K using elbow method
- Debug clustering issues

---

## 📊 What You'll Build

A complete K-Means clustering system that can:
1. **Initialize centroids** (random or K-Means++)
2. **Assign points** to nearest centroid
3. **Update centroids** to mean of assigned points
4. **Iterate until convergence**
5. **Visualize clusters** (2D scatter plots)
6. **Evaluate clustering** (WCSS, silhouette score)
7. **Find optimal K** (elbow method)

---

## 🔢 The Algorithm

### K-Means in Plain English

```
1. Choose K (number of clusters)
2. Initialize K centroids
3. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update each centroid to mean of its points
4. Return final centroids and assignments
```

### The Math

**Distance (Euclidean)**:
```
d(p, q) = √((p₁ - q₁)² + (p₂ - q₂)² + ... + (pₙ - qₙ)²)
```

**Assignment**:
```
cluster(xᵢ) = argmin d(xᵢ, μⱼ)  for j = 1 to K

(Assign xᵢ to nearest centroid μⱼ)
```

**Update**:
```
μⱼ = (1/|Cⱼ|) × Σ xᵢ  for all xᵢ in cluster j

(New centroid = average of all points in cluster)
```

**Convergence**:
```
Stop when: ||μⱼ_new - μⱼ_old|| < ε  for all j

(Centroids stop moving significantly)
```

---

## 📁 Project Structure

```
project-2-kmeans-clustering/
├── src/
│   ├── kmeans.ts              # Main K-Means implementation
│   ├── utils.ts               # Distance, normalization helpers
│   ├── demo.ts                # Demos with various datasets
│   └── visualize.ts           # ASCII visualization
├── data/
│   └── sample-datasets.ts     # Pre-made datasets
├── package.json
├── tsconfig.json
├── k-means-theory.md          # Theory guide
└── README.md                  # This file
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Demo
```bash
npm run demo
```

### 3. Build for Production
```bash
npm run build
node dist/demo.js
```

---

## 💡 Implementation Guide

### Step 1: Create the KMeans Class

```typescript
// src/kmeans.ts

export class KMeans {
  private k: number;
  private maxIterations: number;
  private tolerance: number;
  private centroids: number[][] = [];
  private labels: number[] = [];
  private wcss: number = 0;

  constructor(k: number, maxIterations: number = 100, tolerance: number = 0.0001) {
    this.k = k;
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
  }

  // TODO: Implement these methods
  fit(X: number[][]): void {}
  predict(X: number[][]): number[] {}
  getCentroids(): number[][] {}
  getWCSS(): number {}
}
```

### Step 2: Implement Distance Calculation

```typescript
// src/utils.ts

export function euclideanDistance(p1: number[], p2: number[]): number {
  // TODO: Calculate Euclidean distance
  // Formula: √(Σ(p1ᵢ - p2ᵢ)²)
}

export function normalizeFeatures(X: number[][]): {
  normalized: number[][];
  means: number[];
  stds: number[];
} {
  // TODO: Z-score normalization
  // Important: Clustering sensitive to scale!
}
```

### Step 3: Implement Initialization

```typescript
// In KMeans class

private initializeCentroids(X: number[][]): void {
  // Method 1: Random initialization
  // Pick K random points from X

  // Method 2 (Better): K-Means++
  // 1. Pick first centroid randomly
  // 2. For each next centroid:
  //    - Calculate D(x)² for each point (distance to nearest centroid)
  //    - Pick point with probability proportional to D(x)²
}
```

### Step 4: Implement Assignment Step

```typescript
private assignClusters(X: number[][]): number[] {
  // For each point:
  //   1. Calculate distance to each centroid
  //   2. Assign to nearest centroid
  //   3. Return array of cluster assignments
}
```

### Step 5: Implement Update Step

```typescript
private updateCentroids(X: number[][], labels: number[]): void {
  // For each cluster j:
  //   1. Find all points assigned to cluster j
  //   2. Calculate mean of those points
  //   3. Update μⱼ to that mean
}
```

### Step 6: Implement Main Training Loop

```typescript
fit(X: number[][]): void {
  // 1. Initialize centroids
  // 2. For iteration = 0 to maxIterations:
  //    a. Assign points to clusters
  //    b. Store old centroids
  //    c. Update centroids
  //    d. Check convergence (centroids movement < tolerance)
  //    e. If converged, break
  // 3. Calculate final WCSS
}
```

### Step 7: Implement Prediction

```typescript
predict(X: number[][]): number[] {
  // For each point in X:
  //   Assign to nearest centroid
  // Return cluster assignments
}
```

---

## 📈 Datasets to Try

### 1. Simple 2D Blobs
```typescript
// Three well-separated clusters
const X = [
  // Cluster 1 (bottom-left)
  [1, 1], [2, 1], [1, 2], [2, 2],
  // Cluster 2 (top-left)
  [1, 8], [2, 8], [1, 9], [2, 9],
  // Cluster 3 (right)
  [8, 5], [9, 5], [8, 6], [9, 6]
];

const kmeans = new KMeans(3);
kmeans.fit(X);
```

### 2. Customer Segmentation
```typescript
// [Annual Spending ($1000s), Visit Frequency]
const customers = [
  [50, 100],   // Budget, frequent
  [45, 95],
  [200, 10],   // Big spender, rare
  [180, 12],
  [100, 50],   // Regular customer
  [110, 48]
];

const kmeans = new KMeans(3);
kmeans.fit(customers);
```

### 3. Iris Dataset (Classic)
```typescript
// [Sepal Length, Sepal Width, Petal Length, Petal Width]
// Try K=3 (there are 3 species)
```

---

## 🎨 Visualization

### ASCII Visualization (Simple)
```typescript
// src/visualize.ts

export function visualizeClusters2D(X: number[][], labels: number[], centroids: number[][]) {
  // Create ASCII plot
  // Different symbols for different clusters
  // Show centroids with special marker
}
```

**Example Output**:
```
10|
  |        ○○○
  |       ○ ★ ○
 5|
  | ●●●          ▲▲▲
  |● ★ ●        ▲ ★ ▲
 0|_________________________
  0    5   10   15   20

● = Cluster 1,  ○ = Cluster 2,  ▲ = Cluster 3
★ = Centroids
```

---

## ✅ Success Criteria

Your implementation is complete when:

- [ ] **Algorithm works** on simple 2D data
- [ ] **Convergence** happens (centroids stop moving)
- [ ] **Multiple K values** can be tried
- [ ] **WCSS** calculated correctly (decreases with more clusters)
- [ ] **Visualization** shows clusters clearly
- [ ] **Prediction** works on new data
- [ ] **Feature normalization** implemented
- [ ] **K-Means++** initialization option available
- [ ] **Code documented** with clear comments

---

## 🐛 Debugging Tips

### Problem: Centroids don't converge
**Solution**:
- Check initialization (try K-Means++)
- Increase max iterations
- Verify distance calculation
- Check for empty clusters

### Problem: Poor clustering quality
**Solution**:
- **Normalize features** (different scales?)
- Try different K values
- Run multiple times (different initializations)
- Check if data actually has K clusters

### Problem: Empty cluster
**Solution**:
- Happens when no points assigned to a centroid
- Reinitialize that centroid (pick random point)
- Or reduce K

### Problem: All points in one cluster
**Solution**:
- Features not normalized (one feature dominates)
- K too small
- Poor initialization

---

## 🧪 Testing Your Implementation

### Test 1: Perfect Clusters
```typescript
// Three well-separated groups
const X = [
  [0, 0], [0, 1], [1, 0], [1, 1],       // Cluster 1
  [10, 10], [10, 11], [11, 10], [11, 11], // Cluster 2
  [20, 0], [20, 1], [21, 0], [21, 1]    // Cluster 3
];

const kmeans = new KMeans(3);
kmeans.fit(X);

// Should perfectly separate into 3 clusters
// WCSS should be very low
```

### Test 2: Overlapping Clusters
```typescript
// More realistic - some overlap
const X = [
  [1, 1], [2, 1], [1, 2],
  [3, 3], [4, 3], [3, 4], // Some overlap here
  [6, 6], [7, 6], [6, 7]
];

const kmeans = new KMeans(3);
kmeans.fit(X);

// Should still separate reasonably well
```

### Test 3: Finding Optimal K (Elbow Method)
```typescript
const wcssValues: number[] = [];

for (let k = 1; k <= 10; k++) {
  const kmeans = new KMeans(k);
  kmeans.fit(X);
  wcssValues.push(kmeans.getWCSS());
}

// Plot or print wcssValues
// Look for "elbow" where improvement slows
```

---

## 📚 Additional Challenges

Once you have basic implementation working:

1. **K-Means++ Initialization**
   - Implement smarter centroid initialization
   - Compare with random initialization

2. **Elbow Method**
   - Automate finding optimal K
   - Plot WCSS vs K

3. **Silhouette Score**
   - Calculate silhouette coefficient
   - Use to evaluate clustering quality

4. **Mini-Batch K-Means**
   - Use random sample for each iteration
   - Much faster for large datasets

5. **Visualization**
   - Create web interface with HTML Canvas
   - Animate the clustering process

6. **Handle Edge Cases**
   - Empty clusters (reinitialize centroid)
   - Single point (K=1)
   - More clusters than points (K > n)

---

## 🎯 Interview Questions

After completing this project, you should be able to answer:

1. **What's the difference between supervised and unsupervised learning?**
2. **How does K-Means work step by step?**
3. **What is the time complexity of K-Means?**
4. **Why is feature normalization critical for K-Means?**
5. **What are the limitations of K-Means?**
6. **How do you choose K?**
7. **What's the difference between K-Means and K-NN?**
8. **Explain K-Means++ initialization**
9. **When would you NOT use K-Means?**
10. **What metrics evaluate clustering quality?**

---

## 📖 Resources

### Project Documentation
- **[K-Means Theory Guide](./k-means-theory.md)** ⭐ - Complete conceptual explanation

### Video Resources
- [StatQuest - K-Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [3Blue1Brown - Neural Networks (clustering mentioned)](https://www.youtube.com/watch?v=aircAruvnKk)

### Interactive
- [Visualizing K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

---

## 🚀 Next Steps

After completing this project:
1. Compare with Project 1 (supervised vs unsupervised)
2. Try K-Means on real datasets (Iris, customer data)
3. Write blog post: "My First Unsupervised Learning Algorithm"
4. Deploy visualization to show clustering live
5. Update PROGRESS.md
6. Move to Phase 2: Neural Networks!

---

## 💡 Tips

- **Normalize first!** K-Means is very sensitive to scale
- **Run multiple times** with different initializations
- **Visualize when possible** (even ASCII plots help!)
- **Start simple** (2D data, clear clusters)
- **Compare K values** (try K=2, 3, 4, 5...)
- **Check assumptions** (spherical clusters? similar sizes?)

---

## 🔍 Differences from Linear Regression

| Aspect | Linear Regression | K-Means |
|--------|------------------|---------|
| **Learning Type** | Supervised | Unsupervised |
| **Labels** | Required | None |
| **Goal** | Predict values | Find groups |
| **Output** | Continuous number | Cluster assignment |
| **Optimization** | Gradient descent | Iterative assignment |
| **Convergence** | To global minimum (convex) | To local optimum |

---

**Ready to code?** Start with `src/kmeans.ts` and implement the KMeans class!

Remember: **Read the theory guide first!** Understanding unsupervised learning is crucial before diving into code.
