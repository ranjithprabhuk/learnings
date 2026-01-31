/**
 * Basic MNIST Demo
 *
 * This demo shows the complete workflow:
 * 1. Load MNIST data
 * 2. Build and train a neural network
 * 3. Make predictions
 * 4. Evaluate performance
 */

import { MNISTClassifier } from './mnist-classifier';
import { DataLoader } from './data-loader';
import {
  printTrainingHistory,
  printSummaryStatistics,
  printConfusionMatrix,
  printClassAccuracy,
} from './utils';

async function main() {
  console.log('╔═══════════════════════════════════════════╗');
  console.log('║     MNIST Handwritten Digit Classifier   ║');
  console.log('║              Basic Demo                   ║');
  console.log('╚═══════════════════════════════════════════╝\n');

  // Step 1: Load data
  console.log('Step 1: Loading MNIST dataset...');
  const dataLoader = new DataLoader();
  const data = await dataLoader.loadMNIST();

  // Show some statistics
  const stats = dataLoader.getStatistics(data.trainImages, data.trainLabels);
  console.log(`Training samples: ${stats.numSamples}`);
  console.log(`Image shape: ${stats.imageShape}`);
  console.log('Class distribution:', stats.classDistribution);

  // Visualize a few samples
  console.log('\nSample images from dataset:');
  for (let i = 0; i < 3; i++) {
    const randomIndex = Math.floor(Math.random() * stats.numSamples);
    const sample = dataLoader.getSamples(data.trainImages, data.trainLabels, [randomIndex]);
    const label = sample.labels.argMax(-1).dataSync()[0];

    console.log(`\nSample ${i + 1}:`);
    dataLoader.visualizeImage(sample.images.squeeze(), label);
  }

  // Step 2: Build model
  console.log('\n\nStep 2: Building neural network...');
  const classifier = new MNISTClassifier();
  classifier.buildModel();

  // Step 3: Train model
  console.log('\nStep 3: Training model...');
  console.log('This may take a few minutes...\n');

  const history = await classifier.train(data.trainImages, data.trainLabels, {
    epochs: 10,
    batchSize: 128,
    validationSplit: 0.15,
    verbose: true,
  });

  // Print training history
  printTrainingHistory(history);

  // Step 4: Evaluate on test set
  console.log('\nStep 4: Evaluating on test set...');
  const metrics = await classifier.evaluate(data.testImages, data.testLabels);

  // Print results
  printSummaryStatistics(metrics);
  printClassAccuracy(metrics.classAccuracy);
  printConfusionMatrix(metrics.confusionMatrix);

  // Step 5: Make some predictions
  console.log('\nStep 5: Making predictions on random samples...');

  for (let i = 0; i < 5; i++) {
    const randomIndex = Math.floor(Math.random() * data.testImages.shape[0]);
    const sample = dataLoader.getSamples(data.testImages, data.testLabels, [randomIndex]);
    const actualLabel = sample.labels.argMax(-1).dataSync()[0];

    const prediction = classifier.predict(sample.images.squeeze());

    console.log(`\nSample ${i + 1}:`);
    dataLoader.visualizeImage(sample.images.squeeze(), actualLabel);
    console.log(`Predicted: ${prediction.digit}`);
    console.log(`Actual:    ${actualLabel}`);
    console.log(`Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
    console.log(`Result:    ${prediction.digit === actualLabel ? '✓ Correct' : '✗ Wrong'}`);

    // Show all probabilities
    console.log('All probabilities:');
    prediction.probabilities.forEach((prob, digit) => {
      const bar = '█'.repeat(Math.round(prob * 30));
      console.log(`  ${digit}: ${(prob * 100).toFixed(2)}% ${bar}`);
    });
  }

  // Step 6: Save model
  console.log('\n\nStep 6: Saving trained model...');
  await classifier.save('file://./models/mnist-model');

  console.log('\n╔═══════════════════════════════════════════╗');
  console.log('║              Demo Complete!               ║');
  console.log('╚═══════════════════════════════════════════╝');

  console.log('\nNext steps:');
  console.log('- Run `npm run demo:predict` to test more predictions');
  console.log('- Run `npm run demo:evaluate` for detailed evaluation');
  console.log('- Run `npm run demo:visualize` to visualize results');

  // Cleanup
  classifier.dispose();
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
