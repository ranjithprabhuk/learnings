/**
 * Training Demo
 *
 * Comprehensive training demo with:
 * - Progress tracking
 * - Model checkpointing
 * - Training visualization
 */

import { MNISTClassifier } from './mnist-classifier';
import { DataLoader } from './data-loader';
import { printTrainingHistory, plotTrainingCurves } from './utils';

async function main() {
  console.log('╔═══════════════════════════════════════════╗');
  console.log('║         MNIST Training Pipeline           ║');
  console.log('╚═══════════════════════════════════════════╝\n');

  // Load data
  const dataLoader = new DataLoader();
  const data = await dataLoader.loadMNIST();

  // Build model
  const classifier = new MNISTClassifier();
  classifier.buildModel();

  // Configuration
  const config = {
    epochs: 15,
    batchSize: 128,
    learningRate: 0.001,
    validationSplit: 0.15,
    verbose: true,
  };

  console.log('\n=== Training Configuration ===');
  console.log(`Epochs: ${config.epochs}`);
  console.log(`Batch Size: ${config.batchSize}`);
  console.log(`Learning Rate: ${config.learningRate}`);
  console.log(`Validation Split: ${config.validationSplit}`);
  console.log('===============================\n');

  // Train with progress tracking
  console.log('Starting training...\n');

  const startTime = Date.now();

  const history = await classifier.train(data.trainImages, data.trainLabels, config);

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`\nTraining completed in ${duration}s`);

  // Print summary
  printTrainingHistory(history);

  // Plot training curves
  plotTrainingCurves(history);

  // Evaluate
  console.log('\nEvaluating on test set...');
  const metrics = await classifier.evaluate(data.testImages, data.testLabels);

  console.log('\n=== Final Results ===');
  console.log(`Test Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
  console.log(`Test Loss: ${metrics.loss.toFixed(4)}`);

  // Check if model is good enough
  if (metrics.accuracy >= 0.97) {
    console.log('\n✓ Excellent! Accuracy >= 97%');
  } else if (metrics.accuracy >= 0.95) {
    console.log('\n✓ Good! Accuracy >= 95%');
  } else {
    console.log('\n⚠ Warning: Accuracy < 95%. Consider training longer or adjusting hyperparameters.');
  }

  // Save model
  console.log('\nSaving trained model...');
  await classifier.save('file://./models/mnist-model');
  console.log('Model saved to ./models/mnist-model');

  console.log('\n╔═══════════════════════════════════════════╗');
  console.log('║           Training Complete!              ║');
  console.log('╚═══════════════════════════════════════════╝');

  // Cleanup
  classifier.dispose();
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
