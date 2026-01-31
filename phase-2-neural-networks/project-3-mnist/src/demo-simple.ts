/**
 * Simple Demo with Synthetic Data
 *
 * Demonstrates the neural network without needing to download MNIST
 * Uses synthetic digit-like patterns for training
 */

import * as tf from '@tensorflow/tfjs';
import { MNISTClassifier } from './mnist-classifier';
import { printTrainingHistory, printSummaryStatistics } from './utils';

// Generate synthetic "digit" data
function generateSyntheticData(numSamples: number): { images: tf.Tensor; labels: tf.Tensor } {
  const images: number[][] = [];
  const labels: number[] = [];

  for (let i = 0; i < numSamples; i++) {
    const digit = Math.floor(Math.random() * 10);
    const image = new Array(784).fill(0);

    // Create simple patterns for each digit
    // This is simplified - just demonstrates the concept
    for (let j = 0; j < 784; j++) {
      // Add some pattern based on digit + noise
      const pattern = Math.sin((j + digit * 50) / 10) * 0.5 + 0.5;
      const noise = (Math.random() - 0.5) * 0.2;
      image[j] = Math.max(0, Math.min(1, pattern + noise));
    }

    images.push(image);
    labels.push(digit);
  }

  return {
    images: tf.tensor2d(images),
    labels: tf.oneHot(tf.tensor1d(labels, 'int32'), 10),
  };
}

async function main() {
  console.log('╔═══════════════════════════════════════════╗');
  console.log('║  MNIST Neural Network - Simple Demo      ║');
  console.log('║  (Using Synthetic Data for Testing)      ║');
  console.log('╚═══════════════════════════════════════════╝\n');

  console.log('Note: This demo uses synthetic data to demonstrate');
  console.log('the neural network without needing to download MNIST.\n');

  // Generate training and test data
  console.log('Generating synthetic training data (5000 samples)...');
  const trainData = generateSyntheticData(5000);

  console.log('Generating synthetic test data (1000 samples)...\n');
  const testData = generateSyntheticData(1000);

  // Build model
  console.log('Building neural network...');
  const classifier = new MNISTClassifier();
  classifier.buildModel();

  // Train model
  console.log('\nTraining model (this will take a few minutes)...\n');

  const history = await classifier.train(trainData.images, trainData.labels, {
    epochs: 20,
    batchSize: 128,
    validationSplit: 0.15,
    verbose: true,
  });

  // Print training results
  printTrainingHistory(history);

  // Evaluate on test set
  console.log('\nEvaluating on test set...');
  const metrics = await classifier.evaluate(testData.images, testData.labels);

  printSummaryStatistics(metrics);

  // Make some predictions
  console.log('\n=== Sample Predictions ===\n');

  for (let i = 0; i < 10; i++) {
    const sampleImage = testData.images.slice([i, 1]);
    const actualLabel = testData.labels.slice([i, 1]).argMax(-1).dataSync()[0];

    const prediction = classifier.predict(sampleImage.squeeze());

    console.log(`Sample ${i + 1}:`);
    console.log(`  Predicted: ${prediction.digit}`);
    console.log(`  Actual:    ${actualLabel}`);
    console.log(`  Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
    console.log(`  Result:    ${prediction.digit === actualLabel ? '✓ Correct' : '✗ Wrong'}\n`);
  }

  // Save model
  console.log('\nSaving trained model...');
  await classifier.save('file://./models/mnist-model');
  console.log('Model saved successfully!');

  console.log('\n╔═══════════════════════════════════════════╗');
  console.log('║          Demo Complete!                   ║');
  console.log('╚═══════════════════════════════════════════╝');

  console.log('\nKey Learnings:');
  console.log('✓ Neural network architecture (784→128→64→10)');
  console.log('✓ Forward propagation (making predictions)');
  console.log('✓ Backward propagation (learning from errors)');
  console.log('✓ Training loop with epochs and batches');
  console.log('✓ Model evaluation and metrics');
  console.log('✓ Saving and loading models');

  console.log('\nNote: With real MNIST data, you would see 97-98% accuracy.');
  console.log('This synthetic data demo shows the neural network working!');

  // Cleanup
  classifier.dispose();
  trainData.images.dispose();
  trainData.labels.dispose();
  testData.images.dispose();
  testData.labels.dispose();
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
