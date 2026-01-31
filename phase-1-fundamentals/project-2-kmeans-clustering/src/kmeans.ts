/**
 * K-Means Clustering Implementation from Scratch
 *
 * This implements the K-Means algorithm for unsupervised learning
 * without using any ML libraries.
 *
 * Learning Objectives:
 * - Understand unsupervised learning
 * - Implement iterative clustering algorithm
 * - Learn distance calculations
 * - Master centroid initialization strategies
 */

import { euclideanDistance } from './utils.js';

export class KMeans {
  private k: number;
  private maxIterations: number;
  private tolerance: number;
  private centroids: number[][] = [];
  private labels: number[] = [];
  private wcss: number = 0; // Within-Cluster Sum of Squares
  private initMethod: 'random' | 'kmeans++';

  constructor(
    k: number,
    maxIterations: number = 100,
    tolerance: number = 0.0001,
    initMethod: 'random' | 'kmeans++' = 'kmeans++'
  ) {
    this.k = k;
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
    this.initMethod = initMethod;
  }

  /**
   * Fit the K-Means model to the data
   * @param X - Data matrix (m x n) where m = samples, n = features
   */
  fit(X: number[][]): void {
    if (X.length === 0) {
      throw new Error('Cannot fit on empty dataset');
    }

    if (this.k > X.length) {
      throw new Error(`K (${this.k}) cannot be greater than number of samples (${X.length})`);
    }

    console.log(`\n🎯 K-Means Clustering`);
    console.log(`Samples: ${X.length}, Features: ${X[0].length}`);
    console.log(`K: ${this.k}, Max Iterations: ${this.maxIterations}`);
    console.log(`Initialization: ${this.initMethod}\n`);

    // Initialize centroids
    if (this.initMethod === 'kmeans++') {
      this.initializeCentroidsKMeansPlusPlus(X);
    } else {
      this.initializeCentroidsRandom(X);
    }

    // Main K-Means loop
    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      // Store old centroids to check convergence
      const oldCentroids = this.centroids.map(c => [...c]);

      // Assignment step: assign each point to nearest centroid
      this.labels = this.assignClusters(X);

      // Update step: recalculate centroids
      this.updateCentroids(X, this.labels);

      // Calculate WCSS
      this.wcss = this.calculateWCSS(X, this.labels);

      // Check convergence
      const maxMovement = this.calculateMaxCentroidMovement(oldCentroids, this.centroids);

      if (iteration % 10 === 0 || maxMovement < this.tolerance) {
        console.log(
          `Iteration ${iteration.toString().padStart(3)}: ` +
          `WCSS = ${this.wcss.toFixed(4)}, ` +
          `Max centroid movement = ${maxMovement.toFixed(6)}`
        );
      }

      // Converged if centroids moved less than tolerance
      if (maxMovement < this.tolerance) {
        console.log(`\n✅ Converged after ${iteration + 1} iterations!`);
        console.log(`Final WCSS: ${this.wcss.toFixed(4)}\n`);
        return;
      }
    }

    console.log(`\n⚠️  Reached max iterations (${this.maxIterations})`);
    console.log(`Final WCSS: ${this.wcss.toFixed(4)}\n`);
  }

  /**
   * Predict cluster assignments for new data
   * @param X - Data matrix
   * @returns Array of cluster assignments (0 to k-1)
   */
  predict(X: number[][]): number[] {
    if (this.centroids.length === 0) {
      throw new Error('Model not fitted yet. Call fit() first.');
    }

    return this.assignClusters(X);
  }

  /**
   * Initialize centroids randomly from data points
   * @param X - Data matrix
   */
  private initializeCentroidsRandom(X: number[][]): void {
    // Randomly select K points from X as initial centroids
    const indices = new Set<number>();
    while (indices.size < this.k) {
      indices.add(Math.floor(Math.random() * X.length));
    }

    this.centroids = Array.from(indices).map(i => [...X[i]]);
  }

  /**
   * Initialize centroids using K-Means++ algorithm
   * This tends to give better initial centroids than random selection
   * @param X - Data matrix
   */
  private initializeCentroidsKMeansPlusPlus(X: number[][]): void {
    // Step 1: Choose first centroid randomly
    const firstIndex = Math.floor(Math.random() * X.length);
    this.centroids = [[...X[firstIndex]]];

    // Step 2: For each remaining centroid
    for (let c = 1; c < this.k; c++) {
      // Calculate D(x)² for each point (squared distance to nearest centroid)
      const distances: number[] = [];
      let totalDistance = 0;

      for (const point of X) {
        // Find distance to nearest centroid
        let minDist = Infinity;
        for (const centroid of this.centroids) {
          const dist = euclideanDistance(point, centroid);
          minDist = Math.min(minDist, dist);
        }

        const distSquared = minDist * minDist;
        distances.push(distSquared);
        totalDistance += distSquared;
      }

      // Choose next centroid with probability proportional to D(x)²
      let random = Math.random() * totalDistance;
      for (let i = 0; i < X.length; i++) {
        random -= distances[i];
        if (random <= 0) {
          this.centroids.push([...X[i]]);
          break;
        }
      }
    }
  }

  /**
   * Assign each point to nearest centroid
   * @param X - Data matrix
   * @returns Array of cluster assignments
   */
  private assignClusters(X: number[][]): number[] {
    const labels: number[] = [];

    for (const point of X) {
      let minDistance = Infinity;
      let closestCentroid = 0;

      // Find nearest centroid
      for (let c = 0; c < this.k; c++) {
        const distance = euclideanDistance(point, this.centroids[c]);
        if (distance < minDistance) {
          minDistance = distance;
          closestCentroid = c;
        }
      }

      labels.push(closestCentroid);
    }

    return labels;
  }

  /**
   * Update centroids to mean of assigned points
   * @param X - Data matrix
   * @param labels - Cluster assignments
   */
  private updateCentroids(X: number[][], labels: number[]): void {
    const numFeatures = X[0].length;

    // For each cluster
    for (let c = 0; c < this.k; c++) {
      // Find all points in this cluster
      const clusterPoints: number[][] = [];
      for (let i = 0; i < X.length; i++) {
        if (labels[i] === c) {
          clusterPoints.push(X[i]);
        }
      }

      // Handle empty cluster (reinitialize to random point)
      if (clusterPoints.length === 0) {
        console.warn(`⚠️  Cluster ${c} is empty, reinitializing...`);
        const randomIndex = Math.floor(Math.random() * X.length);
        this.centroids[c] = [...X[randomIndex]];
        continue;
      }

      // Calculate mean of all points in cluster
      const newCentroid: number[] = new Array(numFeatures).fill(0);
      for (const point of clusterPoints) {
        for (let f = 0; f < numFeatures; f++) {
          newCentroid[f] += point[f];
        }
      }

      for (let f = 0; f < numFeatures; f++) {
        newCentroid[f] /= clusterPoints.length;
      }

      this.centroids[c] = newCentroid;
    }
  }

  /**
   * Calculate Within-Cluster Sum of Squares (WCSS)
   * Lower is better - measures compactness of clusters
   * @param X - Data matrix
   * @param labels - Cluster assignments
   * @returns WCSS value
   */
  private calculateWCSS(X: number[][], labels: number[]): number {
    let wcss = 0;

    for (let i = 0; i < X.length; i++) {
      const clusterLabel = labels[i];
      const centroid = this.centroids[clusterLabel];
      const distance = euclideanDistance(X[i], centroid);
      wcss += distance * distance; // Squared distance
    }

    return wcss;
  }

  /**
   * Calculate maximum movement of any centroid
   * Used to check convergence
   * @param oldCentroids - Previous centroids
   * @param newCentroids - Updated centroids
   * @returns Maximum distance any centroid moved
   */
  private calculateMaxCentroidMovement(
    oldCentroids: number[][],
    newCentroids: number[][]
  ): number {
    let maxMovement = 0;

    for (let c = 0; c < this.k; c++) {
      const movement = euclideanDistance(oldCentroids[c], newCentroids[c]);
      maxMovement = Math.max(maxMovement, movement);
    }

    return maxMovement;
  }

  /**
   * Get final centroids
   */
  getCentroids(): number[][] {
    return this.centroids.map(c => [...c]);
  }

  /**
   * Get cluster assignments for training data
   */
  getLabels(): number[] {
    return [...this.labels];
  }

  /**
   * Get final WCSS (Within-Cluster Sum of Squares)
   * Lower is better
   */
  getWCSS(): number {
    return this.wcss;
  }

  /**
   * Get cluster sizes
   */
  getClusterSizes(): number[] {
    const sizes = new Array(this.k).fill(0);
    for (const label of this.labels) {
      sizes[label]++;
    }
    return sizes;
  }
}
