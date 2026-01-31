/**
 * Visualization Demo
 *
 * Visualize:
 * - Model predictions
 * - Confusion patterns
 * - Correct vs incorrect predictions
 */

import { MNISTClassifier } from './mnist-classifier';
import { DataLoader } from './data-loader';
import { printConfusionMatrix } from './utils';

async function main() {
  console.log('╔═══════════════════════════════════════════╗');
  console.log('║      MNIST Visualization Demo             ║');
  console.log('╚═══════════════════════════════════════════╝\n');

  // Load data
  const dataLoader = new DataLoader();
  const data = await dataLoader.loadMNIST();

  // Load model
  const classifier = new MNISTClassifier();

  try {
    await classifier.load('file://./models/mnist-model/model.json');
  } catch (error) {
    console.error('\n❌ Error: Could not load model.');
    console.error('Please run `npm run demo:train` first to train a model.\n');
    process.exit(1);
  }

  console.log('Model loaded successfully!\n');

  // 1. Show correct predictions with high confidence
  console.log('═'.repeat(60));
  console.log('1. CORRECT PREDICTIONS (High Confidence)');
  console.log('═'.repeat(60));

  for (let i = 0; i < 5; i++) {
    let found = false;
    let attempts = 0;

    while (!found && attempts < 100) {
      const randomIndex = Math.floor(Math.random() * data.testImages.shape[0]);
      const sample = dataLoader.getSamples(data.testImages, data.testLabels, [randomIndex]);

      const actualLabel = sample.labels.argMax(-1).dataSync()[0];
      const prediction = classifier.predict(sample.images.squeeze());

      if (prediction.digit === actualLabel && prediction.confidence > 0.99) {
        console.log(`\nExample ${i + 1}: Digit ${actualLabel}`);
        console.log(`Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
        dataLoader.visualizeImage(sample.images.squeeze());

        console.log('Probability distribution:');
        prediction.probabilities.forEach((prob, digit) => {
          if (prob > 0.001) {
            const bar = '█'.repeat(Math.round(prob * 40));
            console.log(`  ${digit}: ${(prob * 100).toFixed(2)}% ${bar}`);
          }
        });

        found = true;
      }
      attempts++;
    }
  }

  // 2. Show correct predictions with low confidence
  console.log('\n' + '═'.repeat(60));
  console.log('2. CORRECT PREDICTIONS (Low Confidence - Model is Uncertain)');
  console.log('═'.repeat(60));

  for (let i = 0; i < 5; i++) {
    let found = false;
    let attempts = 0;

    while (!found && attempts < 100) {
      const randomIndex = Math.floor(Math.random() * data.testImages.shape[0]);
      const sample = dataLoader.getSamples(data.testImages, data.testLabels, [randomIndex]);

      const actualLabel = sample.labels.argMax(-1).dataSync()[0];
      const prediction = classifier.predict(sample.images.squeeze());

      if (prediction.digit === actualLabel && prediction.confidence < 0.7 && prediction.confidence > 0.4) {
        console.log(`\nExample ${i + 1}: Digit ${actualLabel}`);
        console.log(`Confidence: ${(prediction.confidence * 100).toFixed(2)}% (uncertain!)`);
        dataLoader.visualizeImage(sample.images.squeeze());

        console.log('Probability distribution:');
        prediction.probabilities.forEach((prob, digit) => {
          if (prob > 0.05) {
            const bar = '█'.repeat(Math.round(prob * 40));
            console.log(`  ${digit}: ${(prob * 100).toFixed(2)}% ${bar}`);
          }
        });

        found = true;
      }
      attempts++;
    }
  }

  // 3. Show incorrect predictions
  console.log('\n' + '═'.repeat(60));
  console.log('3. INCORRECT PREDICTIONS (Model Mistakes)');
  console.log('═'.repeat(60));

  for (let i = 0; i < 5; i++) {
    let found = false;
    let attempts = 0;

    while (!found && attempts < 1000) {
      const randomIndex = Math.floor(Math.random() * data.testImages.shape[0]);
      const sample = dataLoader.getSamples(data.testImages, data.testLabels, [randomIndex]);

      const actualLabel = sample.labels.argMax(-1).dataSync()[0];
      const prediction = classifier.predict(sample.images.squeeze());

      if (prediction.digit !== actualLabel) {
        console.log(`\nExample ${i + 1}:`);
        console.log(`Actual:    ${actualLabel}`);
        console.log(`Predicted: ${prediction.digit}`);
        console.log(`Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
        dataLoader.visualizeImage(sample.images.squeeze(), actualLabel);

        console.log('Probability distribution:');
        prediction.probabilities.forEach((prob, digit) => {
          if (prob > 0.05) {
            const marker = digit === actualLabel ? '← (actual)' : digit === prediction.digit ? '← (predicted)' : '';
            const bar = '█'.repeat(Math.round(prob * 40));
            console.log(`  ${digit}: ${(prob * 100).toFixed(2)}% ${bar} ${marker}`);
          }
        });

        found = true;
      }
      attempts++;
    }
  }

  // 4. Show confusion matrix
  console.log('\n' + '═'.repeat(60));
  console.log('4. CONFUSION MATRIX');
  console.log('═'.repeat(60));

  const metrics = await classifier.evaluate(data.testImages, data.testLabels);
  printConfusionMatrix(metrics.confusionMatrix);

  // Analysis
  console.log('\n=== Common Confusion Patterns ===');

  // Find most confused pairs
  const confusions: Array<{ from: number; to: number; count: number }> = [];
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) {
      if (i !== j && metrics.confusionMatrix[i][j] > 0) {
        confusions.push({
          from: i,
          to: j,
          count: metrics.confusionMatrix[i][j],
        });
      }
    }
  }

  confusions.sort((a, b) => b.count - a.count);

  console.log('\nMost common confusions:');
  confusions.slice(0, 10).forEach((c, idx) => {
    console.log(`${idx + 1}. ${c.from} confused as ${c.to}: ${c.count} times`);
  });

  console.log('\n=== Analysis ===');
  console.log('Common patterns in mistakes:');
  console.log('- 3 ↔ 5: Similar rounded shapes');
  console.log('- 4 ↔ 9: Similar vertical structure');
  console.log('- 7 ↔ 2: Similar angled strokes');
  console.log('- 8 ↔ 3: Both have multiple curves');

  console.log('\n╔═══════════════════════════════════════════╗');
  console.log('║       Visualization Complete!             ║');
  console.log('╚═══════════════════════════════════════════╝');

  // Cleanup
  classifier.dispose();
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
