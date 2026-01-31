/**
 * Prediction Demo
 *
 * Load a trained model and make predictions on test images
 */

import { MNISTClassifier } from './mnist-classifier';
import { DataLoader } from './data-loader';

async function main() {
  console.log('╔═══════════════════════════════════════════╗');
  console.log('║        MNIST Prediction Demo              ║');
  console.log('╚═══════════════════════════════════════════╝\n');

  // Load data
  console.log('Loading MNIST test data...');
  const dataLoader = new DataLoader();
  const data = await dataLoader.loadMNIST();

  // Load trained model
  console.log('Loading trained model...');
  const classifier = new MNISTClassifier();

  try {
    await classifier.load('file://./models/mnist-model/model.json');
  } catch (error) {
    console.error('\n❌ Error: Could not load model.');
    console.error('Please run `npm run demo:train` first to train a model.\n');
    process.exit(1);
  }

  console.log('Model loaded successfully!\n');

  // Get random test samples
  const numSamples = 20;
  console.log(`Making predictions on ${numSamples} random test images...\n`);

  let correct = 0;
  const predictions: Array<{
    index: number;
    predicted: number;
    actual: number;
    confidence: number;
    correct: boolean;
  }> = [];

  for (let i = 0; i < numSamples; i++) {
    const randomIndex = Math.floor(Math.random() * data.testImages.shape[0]);
    const sample = dataLoader.getSamples(data.testImages, data.testLabels, [randomIndex]);

    const actualLabel = sample.labels.argMax(-1).dataSync()[0];
    const prediction = classifier.predict(sample.images.squeeze());

    const isCorrect = prediction.digit === actualLabel;
    if (isCorrect) correct++;

    predictions.push({
      index: randomIndex,
      predicted: prediction.digit,
      actual: actualLabel,
      confidence: prediction.confidence,
      correct: isCorrect,
    });

    // Display result
    console.log(`Sample ${i + 1} (index ${randomIndex}):`);
    console.log(`  Predicted: ${prediction.digit}`);
    console.log(`  Actual:    ${actualLabel}`);
    console.log(`  Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
    console.log(`  Result:    ${isCorrect ? '✓ Correct' : '✗ Wrong'}`);

    // Show image for wrong predictions
    if (!isCorrect) {
      dataLoader.visualizeImage(sample.images.squeeze(), actualLabel);
      console.log(`  Model thought it was: ${prediction.digit}`);
    }

    console.log();
  }

  // Summary
  console.log('═'.repeat(50));
  console.log(`Accuracy on ${numSamples} samples: ${correct}/${numSamples} (${((correct / numSamples) * 100).toFixed(2)}%)`);
  console.log('═'.repeat(50));

  // Show some interesting cases
  console.log('\n=== Correct Predictions (High Confidence) ===');
  const highConfidence = predictions
    .filter((p) => p.correct)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);

  highConfidence.forEach((p, i) => {
    console.log(
      `${i + 1}. Sample ${p.index}: Predicted ${p.predicted} with ${(p.confidence * 100).toFixed(2)}% confidence ✓`
    );
  });

  if (predictions.some((p) => !p.correct)) {
    console.log('\n=== Incorrect Predictions ===');
    const incorrect = predictions.filter((p) => !p.correct);

    incorrect.forEach((p, i) => {
      console.log(
        `${i + 1}. Sample ${p.index}: Predicted ${p.predicted} (${(p.confidence * 100).toFixed(2)}%), Actually ${p.actual} ✗`
      );
    });

    console.log('\nNote: To see the images of misclassified digits, check the visualization above.');
  }

  console.log('\n╔═══════════════════════════════════════════╗');
  console.log('║          Prediction Demo Complete!        ║');
  console.log('╚═══════════════════════════════════════════╝');

  // Cleanup
  classifier.dispose();
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
