/**
 * Linear Regression Implementation from Scratch
 *
 * This implements gradient descent to learn optimal parameters (theta)
 * for linear regression without using any ML libraries.
 *
 * Learning Objectives:
 * - Understand gradient descent algorithm
 * - Implement cost function (MSE)
 * - Learn matrix operations
 * - Debug ML algorithms
 */

export class LinearRegression {
  private theta: number[] = [];
  private learningRate: number;
  private iterations: number;
  private costHistory: number[] = [];
  private normalize: boolean;
  private featureMeans: number[] = [];
  private featureStdDevs: number[] = [];

  constructor(learningRate: number = 0.01, iterations: number = 1000, normalize: boolean = false) {
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.normalize = normalize;
  }

  /**
   * Train the model using gradient descent
   * @param X - Feature matrix (m x n) where m = samples, n = features
   * @param y - Target values (m x 1)
   */
  fit(X: number[][], y: number[]): void {
    // Normalize features if enabled
    let X_processed = X;
    if (this.normalize) {
      console.log('🔧 Feature normalization enabled...');
      X_processed = this.normalizeFeatures(X, true);
    }

    // Add bias term (column of 1s) to X
    const X_with_bias = this.addBiasTerm(X_processed);
    const m = y.length; // number of training examples
    const n = X_with_bias[0].length; // number of features (including bias)

    // Initialize theta with zeros
    this.theta = new Array(n).fill(0);

    console.log(`\n🚀 Starting training...`);
    console.log(`Samples: ${m}, Features: ${n - 1} (+ bias)`);
    console.log(`Learning rate: ${this.learningRate}, Iterations: ${this.iterations}\n`);

    // Run gradient descent
    this.gradientDescent(X_with_bias, y);

    console.log(`\n✅ Training complete!`);
    console.log(`Final cost: ${this.costHistory[this.costHistory.length - 1].toFixed(6)}`);
    console.log(`Learned parameters (theta): [${this.theta.map(t => t.toFixed(4)).join(', ')}]\n`);
  }

  /**
   * Make predictions using learned parameters
   * @param X - Feature matrix
   * @returns Predicted values
   */
  predict(X: number[][]): number[] {
    // Normalize features if normalization was used during training
    let X_processed = X;
    if (this.normalize) {
      X_processed = this.normalizeFeatures(X, false);
    }

    const X_with_bias = this.addBiasTerm(X_processed);

    // h(x) = X * theta (matrix multiplication)
    const predictions: number[] = [];

    for (const row of X_with_bias) {
      let prediction = 0;
      for (let j = 0; j < this.theta.length; j++) {
        prediction += row[j] * this.theta[j];
      }
      predictions.push(prediction);
    }

    return predictions;
  }

  /**
   * Compute cost function (Mean Squared Error)
   * J(theta) = (1/2m) * sum((h(x) - y)^2)
   *
   * @param X - Feature matrix (with bias term)
   * @param y - Actual values
   * @returns Cost value
   */
  private computeCost(X: number[][], y: number[]): number {
    const m = y.length;
    const predictions = this.predictWithBias(X);

    // Calculate sum of squared errors
    let sumSquaredError = 0;
    for (let i = 0; i < m; i++) {
      const error = predictions[i] - y[i];
      sumSquaredError += error * error;
    }

    // MSE = (1/2m) * sum((prediction - actual)^2)
    const cost = (1 / (2 * m)) * sumSquaredError;

    return cost;
  }

  /**
   * Perform gradient descent optimization
   *
   * Algorithm:
   * Repeat {
   *   theta_j := theta_j - alpha * (1/m) * sum((h(x) - y) * x_j)
   * }
   *
   * @param X - Feature matrix (with bias term already added)
   * @param y - Target values
   */
  private gradientDescent(X: number[][], y: number[]): void {
    const m = y.length;
    const n = this.theta.length;

    for (let iter = 0; iter < this.iterations; iter++) {
      // 1. Calculate predictions
      const predictions = this.predictWithBias(X);

      // 2. Calculate errors
      const errors: number[] = [];
      for (let i = 0; i < m; i++) {
        errors.push(predictions[i] - y[i]);
      }

      // 3. Calculate gradients for each parameter
      const gradients: number[] = new Array(n).fill(0);

      for (let j = 0; j < n; j++) {
        let gradient = 0;
        for (let i = 0; i < m; i++) {
          gradient += errors[i] * X[i][j];
        }
        gradients[j] = (1 / m) * gradient;
      }

      // 4. Update parameters (theta)
      for (let j = 0; j < n; j++) {
        this.theta[j] = this.theta[j] - this.learningRate * gradients[j];
      }

      // 5. Calculate and store cost
      const cost = this.computeCost(X, y);
      this.costHistory.push(cost);

      // Log progress every 100 iterations
      if (iter % 100 === 0 || iter === this.iterations - 1) {
        console.log(
          `Iteration ${iter.toString().padStart(4)}: Cost = ${cost.toFixed(6)}, ` +
          `Theta = [${this.theta.map(t => t.toFixed(4)).join(', ')}]`
        );
      }
    }
  }

  /**
   * Normalize features using standardization (z-score normalization)
   * Formula: x_norm = (x - mean) / std_dev
   *
   * @param X - Feature matrix to normalize
   * @param isFit - If true, calculate and store mean/std; if false, use stored values
   * @returns Normalized feature matrix
   */
  private normalizeFeatures(X: number[][], isFit: boolean): number[][] {
    const m = X.length;
    const n = X[0].length;

    if (isFit) {
      // Calculate mean and standard deviation for each feature
      this.featureMeans = [];
      this.featureStdDevs = [];

      for (let j = 0; j < n; j++) {
        // Calculate mean
        let sum = 0;
        for (let i = 0; i < m; i++) {
          sum += X[i][j];
        }
        const mean = sum / m;
        this.featureMeans.push(mean);

        // Calculate standard deviation
        let squaredDiffSum = 0;
        for (let i = 0; i < m; i++) {
          squaredDiffSum += Math.pow(X[i][j] - mean, 2);
        }
        const stdDev = Math.sqrt(squaredDiffSum / m);
        this.featureStdDevs.push(stdDev === 0 ? 1 : stdDev); // Avoid division by zero

        console.log(`  Feature ${j}: mean = ${mean.toFixed(2)}, std = ${stdDev.toFixed(2)}`);
      }
    }

    // Normalize the features
    const X_normalized: number[][] = [];
    for (let i = 0; i < m; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        const normalized = (X[i][j] - this.featureMeans[j]) / this.featureStdDevs[j];
        row.push(normalized);
      }
      X_normalized.push(row);
    }

    return X_normalized;
  }

  /**
   * Helper: Add bias term (column of 1s) to feature matrix
   * @param X - Original feature matrix
   * @returns Feature matrix with bias term
   */
  private addBiasTerm(X: number[][]): number[][] {
    return X.map(row => [1, ...row]);
  }

  /**
   * Helper: Predict with bias term already added
   * @param X - Feature matrix with bias
   * @returns Predictions
   */
  private predictWithBias(X: number[][]): number[] {
    const predictions: number[] = [];

    for (const row of X) {
      let prediction = 0;
      for (let j = 0; j < this.theta.length; j++) {
        prediction += row[j] * this.theta[j];
      }
      predictions.push(prediction);
    }

    return predictions;
  }

  /**
   * Get the learned parameters
   */
  getTheta(): number[] {
    return this.theta;
  }

  /**
   * Get the cost history during training
   */
  getCostHistory(): number[] {
    return this.costHistory;
  }

  /**
   * Calculate R² score (coefficient of determination)
   * R² = 1 - (SS_res / SS_tot)
   * where SS_res = sum((y - y_pred)^2)
   *       SS_tot = sum((y - y_mean)^2)
   *
   * R² = 1.0 means perfect fit
   * R² = 0.0 means model is no better than mean
   */
  score(X: number[][], y: number[]): number {
    const predictions = this.predict(X); // predict() handles normalization internally
    const yMean = y.reduce((sum, val) => sum + val, 0) / y.length;

    let ssRes = 0; // Residual sum of squares
    let ssTot = 0; // Total sum of squares

    for (let i = 0; i < y.length; i++) {
      ssRes += Math.pow(y[i] - predictions[i], 2);
      ssTot += Math.pow(y[i] - yMean, 2);
    }

    return 1 - (ssRes / ssTot);
  }
}
