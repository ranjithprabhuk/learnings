/**
 * Utility Functions for K-Means Clustering
 *
 * Distance calculations, normalization, and helper functions
 */

/**
 * Calculate Euclidean distance between two points
 * @param p1 - First point
 * @param p2 - Second point
 * @returns Euclidean distance
 */
export function euclideanDistance(p1: number[], p2: number[]): number {
  if (p1.length !== p2.length) {
    throw new Error('Points must have same dimensionality');
  }

  let sum = 0;
  for (let i = 0; i < p1.length; i++) {
    const diff = p1[i] - p2[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

/**
 * Normalize features using z-score normalization
 * IMPORTANT: Always normalize before clustering!
 * @param X - Data matrix
 * @returns Normalized data, means, and standard deviations
 */
export function normalizeFeatures(X: number[][]): {
  normalized: number[][];
  means: number[];
  stds: number[];
} {
  const numFeatures = X[0].length;
  const means: number[] = [];
  const stds: number[] = [];

  // Calculate mean for each feature
  for (let f = 0; f < numFeatures; f++) {
    let sum = 0;
    for (const point of X) {
      sum += point[f];
    }
    means.push(sum / X.length);
  }

  // Calculate standard deviation for each feature
  for (let f = 0; f < numFeatures; f++) {
    let sumSquaredDiff = 0;
    for (const point of X) {
      const diff = point[f] - means[f];
      sumSquaredDiff += diff * diff;
    }
    const std = Math.sqrt(sumSquaredDiff / X.length);
    stds.push(std === 0 ? 1 : std); // Avoid division by zero
  }

  // Normalize data
  const normalized: number[][] = [];
  for (const point of X) {
    const normalizedPoint: number[] = [];
    for (let f = 0; f < numFeatures; f++) {
      normalizedPoint.push((point[f] - means[f]) / stds[f]);
    }
    normalized.push(normalizedPoint);
  }

  return { normalized, means, stds };
}

/**
 * Denormalize a single point (reverse z-score normalization)
 * @param point - Normalized point
 * @param means - Feature means
 * @param stds - Feature standard deviations
 * @returns Original scale point
 */
export function denormalizePoint(point: number[], means: number[], stds: number[]): number[] {
  return point.map((val, i) => val * stds[i] + means[i]);
}

/**
 * Generate random 2D blobs for testing
 * @param numClusters - Number of clusters
 * @param pointsPerCluster - Points per cluster
 * @param spread - Spread of points within cluster
 * @returns Generated data
 */
export function generateBlobs(
  numClusters: number,
  pointsPerCluster: number,
  spread: number = 1.0
): number[][] {
  const data: number[][] = [];

  for (let c = 0; c < numClusters; c++) {
    // Random center for this cluster
    const centerX = Math.random() * 20;
    const centerY = Math.random() * 20;

    // Generate points around this center
    for (let p = 0; p < pointsPerCluster; p++) {
      const x = centerX + (Math.random() - 0.5) * spread * 2;
      const y = centerY + (Math.random() - 0.5) * spread * 2;
      data.push([x, y]);
    }
  }

  return data;
}

/**
 * Shuffle array in place (Fisher-Yates algorithm)
 * @param array - Array to shuffle
 */
export function shuffleArray<T>(array: T[]): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

/**
 * Calculate silhouette coefficient for a clustering
 * Measures how well points are clustered
 * Range: -1 to 1 (higher is better)
 * @param X - Data matrix
 * @param labels - Cluster assignments
 * @returns Average silhouette score
 */
export function silhouetteScore(X: number[][], labels: number[]): number {
  const n = X.length;
  const k = Math.max(...labels) + 1;

  // Group points by cluster
  const clusters: number[][][] = Array(k).fill(null).map(() => []);
  for (let i = 0; i < n; i++) {
    clusters[labels[i]].push(X[i]);
  }

  let totalScore = 0;

  for (let i = 0; i < n; i++) {
    const point = X[i];
    const cluster = labels[i];
    const clusterPoints = clusters[cluster];

    // Skip if cluster has only one point
    if (clusterPoints.length === 1) {
      continue;
    }

    // a: Mean distance to points in same cluster
    let a = 0;
    for (const otherPoint of clusterPoints) {
      if (point !== otherPoint) {
        a += euclideanDistance(point, otherPoint);
      }
    }
    a /= (clusterPoints.length - 1);

    // b: Mean distance to points in nearest other cluster
    let b = Infinity;
    for (let otherCluster = 0; otherCluster < k; otherCluster++) {
      if (otherCluster === cluster) continue;

      let avgDist = 0;
      for (const otherPoint of clusters[otherCluster]) {
        avgDist += euclideanDistance(point, otherPoint);
      }
      avgDist /= clusters[otherCluster].length;

      b = Math.min(b, avgDist);
    }

    // Silhouette coefficient for this point
    const s = (b - a) / Math.max(a, b);
    totalScore += s;
  }

  return totalScore / n;
}
