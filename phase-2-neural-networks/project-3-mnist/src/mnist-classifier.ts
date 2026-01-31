import * as tf from '@tensorflow/tfjs';

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate?: number;
  validationSplit?: number;
  verbose?: boolean;
}

export interface TrainingHistory {
  loss: number[];
  accuracy: number[];
  valLoss: number[];
  valAccuracy: number[];
}

export interface EvaluationMetrics {
  accuracy: number;
  loss: number;
  confusionMatrix: number[][];
  classAccuracy: number[];
  predictions: number[];
  labels: number[];
}

/**
 * MNIST Handwritten Digit Classifier
 *
 * A feedforward neural network for classifying handwritten digits (0-9)
 * from the MNIST dataset.
 *
 * Architecture:
 * - Input: 784 neurons (28x28 flattened image)
 * - Hidden Layer 1: 128 neurons + ReLU
 * - Dropout: 0.2
 * - Hidden Layer 2: 64 neurons + ReLU
 * - Dropout: 0.2
 * - Output: 10 neurons + Softmax
 */
export class MNISTClassifier {
  private model: tf.LayersModel | null = null;
  private readonly inputShape = [784]; // 28x28 flattened
  private readonly numClasses = 10; // Digits 0-9

  /**
   * Build the neural network architecture
   */
  buildModel(): void {
    console.log('Building model architecture...');

    this.model = tf.sequential({
      layers: [
        // Input layer (flattened 28x28 image)
        tf.layers.dense({
          inputShape: this.inputShape,
          units: 128,
          activation: 'relu',
          kernelInitializer: 'heNormal',
          name: 'hidden1',
        }),

        // Dropout for regularization
        tf.layers.dropout({
          rate: 0.2,
          name: 'dropout1',
        }),

        // Second hidden layer
        tf.layers.dense({
          units: 64,
          activation: 'relu',
          kernelInitializer: 'heNormal',
          name: 'hidden2',
        }),

        // Dropout for regularization
        tf.layers.dropout({
          rate: 0.2,
          name: 'dropout2',
        }),

        // Output layer (10 classes)
        tf.layers.dense({
          units: this.numClasses,
          activation: 'softmax',
          kernelInitializer: 'heNormal',
          name: 'output',
        }),
      ],
    });

    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    // Print model summary
    this.model.summary();

    console.log('Model built successfully!');
  }

  /**
   * Train the model on MNIST data
   */
  async train(
    trainImages: tf.Tensor,
    trainLabels: tf.Tensor,
    config: TrainingConfig
  ): Promise<TrainingHistory> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }

    console.log('\n=== Training Configuration ===');
    console.log(`Epochs: ${config.epochs}`);
    console.log(`Batch Size: ${config.batchSize}`);
    console.log(`Learning Rate: ${config.learningRate || 0.001}`);
    console.log(`Validation Split: ${config.validationSplit || 0.0}`);
    console.log('==============================\n');

    const history: TrainingHistory = {
      loss: [],
      accuracy: [],
      valLoss: [],
      valAccuracy: [],
    };

    // Train the model
    const startTime = Date.now();

    const result = await this.model.fit(trainImages, trainLabels, {
      epochs: config.epochs,
      batchSize: config.batchSize,
      validationSplit: config.validationSplit || 0.15,
      verbose: config.verbose !== false ? 1 : 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          history.loss.push(logs?.loss || 0);
          history.accuracy.push(logs?.acc || 0);
          history.valLoss.push(logs?.val_loss || 0);
          history.valAccuracy.push(logs?.val_acc || 0);

          if (config.verbose !== false) {
            console.log(
              `Epoch ${epoch + 1}/${config.epochs} - ` +
              `loss: ${logs?.loss.toFixed(4)} - ` +
              `acc: ${logs?.acc.toFixed(4)} - ` +
              `val_loss: ${logs?.val_loss.toFixed(4)} - ` +
              `val_acc: ${logs?.val_acc.toFixed(4)}`
            );
          }
        },
      },
    });

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`\nTraining completed in ${duration}s`);
    console.log(`Final training accuracy: ${(history.accuracy[history.accuracy.length - 1] * 100).toFixed(2)}%`);
    console.log(`Final validation accuracy: ${(history.valAccuracy[history.valAccuracy.length - 1] * 100).toFixed(2)}%`);

    return history;
  }

  /**
   * Make a prediction on a single image
   * Returns the predicted digit (0-9)
   */
  predict(image: tf.Tensor): { digit: number; confidence: number; probabilities: number[] } {
    if (!this.model) {
      throw new Error('Model not built or loaded. Call buildModel() or load() first.');
    }

    return tf.tidy(() => {
      // Ensure image is the right shape [1, 784]
      let input = image;
      if (input.shape.length === 1) {
        input = input.expandDims(0);
      } else if (input.shape.length === 3 && input.shape[0] === 1) {
        input = input.reshape([1, 784]);
      }

      // Make prediction
      const prediction = this.model!.predict(input) as tf.Tensor;
      const probabilities = Array.from(prediction.dataSync());

      // Get the digit with highest probability
      const digit = prediction.argMax(-1).dataSync()[0];
      const confidence = probabilities[digit];

      return {
        digit,
        confidence,
        probabilities,
      };
    });
  }

  /**
   * Make predictions on multiple images
   */
  predictBatch(images: tf.Tensor): number[] {
    if (!this.model) {
      throw new Error('Model not built or loaded. Call buildModel() or load() first.');
    }

    return tf.tidy(() => {
      const predictions = this.model!.predict(images) as tf.Tensor;
      return Array.from(predictions.argMax(-1).dataSync());
    });
  }

  /**
   * Evaluate model performance on test data
   */
  async evaluate(
    testImages: tf.Tensor,
    testLabels: tf.Tensor
  ): Promise<EvaluationMetrics> {
    if (!this.model) {
      throw new Error('Model not built or loaded. Call buildModel() or load() first.');
    }

    console.log('\n=== Evaluating Model ===');

    // Get predictions
    const predictions = this.predictBatch(testImages);

    // Convert one-hot labels to integers
    const labels = Array.from(testLabels.argMax(-1).dataSync());

    // Calculate overall accuracy
    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === labels[i]) {
        correct++;
      }
    }
    const accuracy = correct / predictions.length;

    // Calculate loss
    const evaluation = await this.model.evaluate(testImages, testLabels) as tf.Scalar[];
    const loss = (await evaluation[0].data())[0];

    // Create confusion matrix
    const confusionMatrix = this.createConfusionMatrix(predictions, labels);

    // Calculate per-class accuracy
    const classAccuracy = this.calculateClassAccuracy(confusionMatrix);

    console.log(`Test Accuracy: ${(accuracy * 100).toFixed(2)}%`);
    console.log(`Test Loss: ${loss.toFixed(4)}`);

    return {
      accuracy,
      loss,
      confusionMatrix,
      classAccuracy,
      predictions,
      labels,
    };
  }

  /**
   * Create confusion matrix
   */
  private createConfusionMatrix(predictions: number[], labels: number[]): number[][] {
    const matrix: number[][] = Array(this.numClasses)
      .fill(0)
      .map(() => Array(this.numClasses).fill(0));

    for (let i = 0; i < predictions.length; i++) {
      matrix[labels[i]][predictions[i]]++;
    }

    return matrix;
  }

  /**
   * Calculate per-class accuracy from confusion matrix
   */
  private calculateClassAccuracy(confusionMatrix: number[][]): number[] {
    const classAccuracy: number[] = [];

    for (let i = 0; i < this.numClasses; i++) {
      const total = confusionMatrix[i].reduce((sum, val) => sum + val, 0);
      const correct = confusionMatrix[i][i];
      classAccuracy.push(total > 0 ? correct / total : 0);
    }

    return classAccuracy;
  }

  /**
   * Save the trained model
   */
  async save(path: string): Promise<void> {
    if (!this.model) {
      throw new Error('Model not built or loaded.');
    }

    console.log(`Saving model to ${path}...`);
    await this.model.save(path);
    console.log('Model saved successfully!');
  }

  /**
   * Load a trained model
   */
  async load(path: string): Promise<void> {
    console.log(`Loading model from ${path}...`);
    this.model = await tf.loadLayersModel(path);
    console.log('Model loaded successfully!');
  }

  /**
   * Get the model (for advanced usage)
   */
  getModel(): tf.LayersModel | null {
    return this.model;
  }

  /**
   * Dispose the model to free memory
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}
