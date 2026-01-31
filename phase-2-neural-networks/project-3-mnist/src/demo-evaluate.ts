/**
 * Evaluation Demo
 *
 * Comprehensive model evaluation with:
 * - Overall metrics
 * - Per-class accuracy
 * - Confusion matrix
 * - Worst predictions
 */

import { MNISTClassifier } from './mnist-classifier';
import { DataLoader } from './data-loader';
import {
  printSummaryStatistics,
  printClassAccuracy,
  printConfusionMatrix,
  printWorstPredictions,
  generateEvaluationReport,
} from './utils';
import * as fs from 'fs';

async function main() {
  console.log('╔═══════════════════════════════════════════╗');
  console.log('║       MNIST Evaluation Report             ║');
  console.log('╚═══════════════════════════════════════════╝\n');

  // Load data
  console.log('Loading MNIST test data...');
  const dataLoader = new DataLoader();
  const data = await dataLoader.loadMNIST();

  // Load model
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

  // Evaluate on test set
  console.log('Evaluating model on 10,000 test images...');
  console.log('This may take a minute...\n');

  const metrics = await classifier.evaluate(data.testImages, data.testLabels);

  // Print detailed report
  printSummaryStatistics(metrics);
  printClassAccuracy(metrics.classAccuracy);
  printConfusionMatrix(metrics.confusionMatrix);
  printWorstPredictions(metrics, 15);

  // Generate and print formatted report
  const report = generateEvaluationReport(metrics);
  console.log('\n' + report);

  // Save report to file
  const reportPath = './evaluation-report.txt';
  console.log(`\nSaving detailed report to ${reportPath}...`);

  const detailedReport = [
    'MNIST Classifier Evaluation Report',
    '='.repeat(60),
    `Generated: ${new Date().toISOString()}`,
    '',
    '1. Overall Performance',
    '─'.repeat(60),
    `Test Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`,
    `Test Loss: ${metrics.loss.toFixed(4)}`,
    `Total Samples: ${metrics.predictions.length}`,
    `Correct Predictions: ${metrics.predictions.filter((p, i) => p === metrics.labels[i]).length}`,
    `Incorrect Predictions: ${metrics.predictions.filter((p, i) => p !== metrics.labels[i]).length}`,
    '',
    '2. Per-Class Accuracy',
    '─'.repeat(60),
  ];

  for (let i = 0; i < metrics.classAccuracy.length; i++) {
    detailedReport.push(`Digit ${i}: ${(metrics.classAccuracy[i] * 100).toFixed(2)}%`);
  }

  detailedReport.push('');
  detailedReport.push('3. Confusion Matrix');
  detailedReport.push('─'.repeat(60));
  detailedReport.push('Rows = True labels, Columns = Predicted labels');
  detailedReport.push('');
  detailedReport.push('     ' + Array.from({ length: 10 }, (_, i) => i.toString().padStart(5)).join(' '));

  for (let i = 0; i < metrics.confusionMatrix.length; i++) {
    const row = metrics.confusionMatrix[i].map((val) => val.toString().padStart(5)).join(' ');
    detailedReport.push(`${i} │ ${row}`);
  }

  fs.writeFileSync(reportPath, detailedReport.join('\n'));
  console.log('Report saved successfully!');

  // Performance analysis
  console.log('\n=== Performance Analysis ===');

  // Check if performance is acceptable
  if (metrics.accuracy >= 0.98) {
    console.log('✓ Excellent performance! (≥98%)');
  } else if (metrics.accuracy >= 0.95) {
    console.log('✓ Good performance! (≥95%)');
  } else if (metrics.accuracy >= 0.90) {
    console.log('⚠ Acceptable performance, but room for improvement (≥90%)');
  } else {
    console.log('✗ Poor performance. Consider:');
    console.log('  - Training for more epochs');
    console.log('  - Adjusting learning rate');
    console.log('  - Adding more layers or neurons');
    console.log('  - Using data augmentation');
  }

  // Check class balance
  const minAcc = Math.min(...metrics.classAccuracy);
  const maxAcc = Math.max(...metrics.classAccuracy);
  const accSpread = maxAcc - minAcc;

  console.log(`\nClass accuracy spread: ${(accSpread * 100).toFixed(2)}%`);
  if (accSpread < 0.05) {
    console.log('✓ Well-balanced performance across all classes');
  } else {
    console.log('⚠ Some classes perform significantly worse than others');
    console.log('  Consider class-specific improvements or data augmentation');
  }

  // Identify problem classes
  const problemClasses = metrics.classAccuracy
    .map((acc, digit) => ({ digit, acc }))
    .filter((c) => c.acc < 0.95)
    .sort((a, b) => a.acc - b.acc);

  if (problemClasses.length > 0) {
    console.log('\n=== Problem Classes (Accuracy < 95%) ===');
    problemClasses.forEach((c) => {
      console.log(`  Digit ${c.digit}: ${(c.acc * 100).toFixed(2)}%`);
    });
  }

  console.log('\n╔═══════════════════════════════════════════╗');
  console.log('║        Evaluation Complete!               ║');
  console.log('╚═══════════════════════════════════════════╝');

  // Cleanup
  classifier.dispose();
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
