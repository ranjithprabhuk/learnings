import { TrainingHistory, EvaluationMetrics } from './mnist-classifier';

/**
 * Utility functions for metrics, visualization, and analysis
 */

/**
 * Print confusion matrix in a readable format
 */
export function printConfusionMatrix(matrix: number[][]): void {
  console.log('\n=== Confusion Matrix ===');
  console.log('Rows = True labels, Columns = Predicted labels\n');

  // Print header
  console.log('    ', ...Array.from({ length: 10 }, (_, i) => i.toString().padStart(5)));
  console.log('    ', '─'.repeat(55));

  // Print each row
  for (let i = 0; i < matrix.length; i++) {
    const row = matrix[i].map((val) => val.toString().padStart(5)).join(' ');
    console.log(`${i} │ ${row}`);
  }
  console.log();
}

/**
 * Print per-class accuracy
 */
export function printClassAccuracy(classAccuracy: number[]): void {
  console.log('\n=== Per-Class Accuracy ===');
  for (let i = 0; i < classAccuracy.length; i++) {
    const acc = (classAccuracy[i] * 100).toFixed(2);
    const bar = '█'.repeat(Math.round(classAccuracy[i] * 50));
    console.log(`Digit ${i}: ${acc}% ${bar}`);
  }
  console.log();
}

/**
 * Print training history summary
 */
export function printTrainingHistory(history: TrainingHistory): void {
  console.log('\n=== Training History ===');

  const finalTrainAcc = (history.accuracy[history.accuracy.length - 1] * 100).toFixed(2);
  const finalValAcc = (history.valAccuracy[history.valAccuracy.length - 1] * 100).toFixed(2);
  const finalTrainLoss = history.loss[history.loss.length - 1].toFixed(4);
  const finalValLoss = history.valLoss[history.valLoss.length - 1].toFixed(4);

  console.log(`Final Training Accuracy:   ${finalTrainAcc}%`);
  console.log(`Final Validation Accuracy: ${finalValAcc}%`);
  console.log(`Final Training Loss:       ${finalTrainLoss}`);
  console.log(`Final Validation Loss:     ${finalValLoss}`);

  // Check for overfitting
  const overfit = Math.abs(history.accuracy[history.accuracy.length - 1] - history.valAccuracy[history.valAccuracy.length - 1]);
  if (overfit > 0.05) {
    console.log(`\n⚠️  Warning: Possible overfitting detected (gap: ${(overfit * 100).toFixed(2)}%)`);
    console.log('   Consider: more dropout, more data, or early stopping');
  } else {
    console.log('\n✓ No significant overfitting detected');
  }

  console.log();
}

/**
 * Plot training curves (ASCII art)
 */
export function plotTrainingCurves(history: TrainingHistory): void {
  console.log('\n=== Training Curves ===\n');

  // Plot loss
  console.log('Loss:');
  plotLine(history.loss, 'Training Loss', 40, 10);
  plotLine(history.valLoss, 'Validation Loss', 40, 10);

  // Plot accuracy
  console.log('\nAccuracy:');
  plotLine(history.accuracy, 'Training Accuracy', 40, 10);
  plotLine(history.valAccuracy, 'Validation Accuracy', 40, 10);

  console.log();
}

/**
 * Plot a single line as ASCII art
 */
function plotLine(data: number[], label: string, width: number, height: number): void {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  // Normalize data to [0, height]
  const normalized = data.map((val) => Math.round(((val - min) / range) * (height - 1)));

  // Create plot
  const plot: string[][] = Array(height)
    .fill(0)
    .map(() => Array(width).fill(' '));

  // Plot points
  const step = data.length / width;
  for (let x = 0; x < width; x++) {
    const dataIndex = Math.min(Math.floor(x * step), data.length - 1);
    const y = height - 1 - normalized[dataIndex];
    plot[y][x] = '█';
  }

  // Print plot
  console.log(`${label} (${min.toFixed(4)} to ${max.toFixed(4)})`);
  for (let y = 0; y < height; y++) {
    console.log('│' + plot[y].join(''));
  }
  console.log('└' + '─'.repeat(width));
  console.log(' ' + ' '.repeat(Math.floor(width / 2) - 3) + 'Epochs');
}

/**
 * Find and print worst predictions
 */
export function printWorstPredictions(
  metrics: EvaluationMetrics,
  numSamples: number = 10
): void {
  console.log(`\n=== Worst ${numSamples} Predictions ===`);

  // Find misclassified samples
  const mistakes: Array<{ index: number; predicted: number; actual: number }> = [];

  for (let i = 0; i < metrics.predictions.length; i++) {
    if (metrics.predictions[i] !== metrics.labels[i]) {
      mistakes.push({
        index: i,
        predicted: metrics.predictions[i],
        actual: metrics.labels[i],
      });
    }
  }

  console.log(`Total mistakes: ${mistakes.length} / ${metrics.predictions.length} (${((mistakes.length / metrics.predictions.length) * 100).toFixed(2)}%)\n`);

  // Print first numSamples mistakes
  const samplesToShow = Math.min(numSamples, mistakes.length);
  for (let i = 0; i < samplesToShow; i++) {
    const mistake = mistakes[i];
    console.log(
      `${i + 1}. Sample ${mistake.index}: ` +
      `Predicted ${mistake.predicted}, Actually ${mistake.actual}`
    );
  }

  console.log();
}

/**
 * Calculate and print summary statistics
 */
export function printSummaryStatistics(metrics: EvaluationMetrics): void {
  console.log('\n=== Summary Statistics ===');

  // Overall accuracy
  console.log(`Overall Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
  console.log(`Overall Loss: ${metrics.loss.toFixed(4)}`);

  // Best and worst performing classes
  const avgClassAcc = metrics.classAccuracy.reduce((a, b) => a + b) / metrics.classAccuracy.length;
  const bestClass = metrics.classAccuracy.indexOf(Math.max(...metrics.classAccuracy));
  const worstClass = metrics.classAccuracy.indexOf(Math.min(...metrics.classAccuracy));

  console.log(`\nAverage Class Accuracy: ${(avgClassAcc * 100).toFixed(2)}%`);
  console.log(`Best Performing Class: ${bestClass} (${(metrics.classAccuracy[bestClass] * 100).toFixed(2)}%)`);
  console.log(`Worst Performing Class: ${worstClass} (${(metrics.classAccuracy[worstClass] * 100).toFixed(2)}%)`);

  // Confusion analysis
  console.log('\nMost Common Confusions:');
  const confusions = findMostCommonConfusions(metrics.confusionMatrix, 5);
  confusions.forEach(({ true: trueClass, predicted: predClass, count }, i) => {
    console.log(`${i + 1}. ${trueClass} misclassified as ${predClass}: ${count} times`);
  });

  console.log();
}

/**
 * Find most common confusions in confusion matrix
 */
function findMostCommonConfusions(
  matrix: number[][],
  topN: number
): Array<{ true: number; predicted: number; count: number }> {
  const confusions: Array<{ true: number; predicted: number; count: number }> = [];

  // Find all off-diagonal elements (mistakes)
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      if (i !== j && matrix[i][j] > 0) {
        confusions.push({
          true: i,
          predicted: j,
          count: matrix[i][j],
        });
      }
    }
  }

  // Sort by count and return top N
  confusions.sort((a, b) => b.count - a.count);
  return confusions.slice(0, topN);
}

/**
 * Create a formatted evaluation report
 */
export function generateEvaluationReport(metrics: EvaluationMetrics): string {
  const lines: string[] = [];

  lines.push('╔═══════════════════════════════════════════╗');
  lines.push('║       MNIST CLASSIFIER EVALUATION        ║');
  lines.push('╚═══════════════════════════════════════════╝');
  lines.push('');

  // Overall metrics
  lines.push('Overall Performance:');
  lines.push(`  Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
  lines.push(`  Loss:     ${metrics.loss.toFixed(4)}`);
  lines.push('');

  // Per-class performance
  lines.push('Per-Class Accuracy:');
  for (let i = 0; i < metrics.classAccuracy.length; i++) {
    const acc = (metrics.classAccuracy[i] * 100).toFixed(2);
    const bar = '█'.repeat(Math.round(metrics.classAccuracy[i] * 30));
    lines.push(`  Digit ${i}: ${acc.padStart(6)}% ${bar}`);
  }
  lines.push('');

  // Confusion analysis
  lines.push('Most Common Confusions:');
  const confusions = findMostCommonConfusions(metrics.confusionMatrix, 5);
  confusions.forEach(({ true: trueClass, predicted: predClass, count }, i) => {
    lines.push(`  ${i + 1}. ${trueClass} → ${predClass}: ${count} times`);
  });
  lines.push('');

  // Total mistakes
  const totalMistakes = metrics.predictions.filter(
    (pred, i) => pred !== metrics.labels[i]
  ).length;
  lines.push(`Total Mistakes: ${totalMistakes} / ${metrics.predictions.length}`);
  lines.push('');

  return lines.join('\n');
}

/**
 * Save evaluation report to a string for file output
 */
export function formatReportForFile(
  metrics: EvaluationMetrics,
  history?: TrainingHistory
): string {
  const lines: string[] = [];

  lines.push('MNIST Classifier Evaluation Report');
  lines.push('='.repeat(50));
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push('');

  // Training history
  if (history) {
    lines.push('Training History:');
    lines.push(`  Epochs: ${history.loss.length}`);
    lines.push(`  Final Training Accuracy: ${(history.accuracy[history.accuracy.length - 1] * 100).toFixed(2)}%`);
    lines.push(`  Final Validation Accuracy: ${(history.valAccuracy[history.valAccuracy.length - 1] * 100).toFixed(2)}%`);
    lines.push('');
  }

  // Evaluation metrics
  lines.push('Test Set Performance:');
  lines.push(`  Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
  lines.push(`  Loss: ${metrics.loss.toFixed(4)}`);
  lines.push('');

  // Confusion matrix
  lines.push('Confusion Matrix:');
  lines.push('     ' + Array.from({ length: 10 }, (_, i) => i.toString().padStart(5)).join(' '));
  for (let i = 0; i < metrics.confusionMatrix.length; i++) {
    const row = metrics.confusionMatrix[i].map((val) => val.toString().padStart(5)).join(' ');
    lines.push(`${i} │ ${row}`);
  }
  lines.push('');

  // Per-class accuracy
  lines.push('Per-Class Accuracy:');
  for (let i = 0; i < metrics.classAccuracy.length; i++) {
    lines.push(`  Digit ${i}: ${(metrics.classAccuracy[i] * 100).toFixed(2)}%`);
  }
  lines.push('');

  return lines.join('\n');
}
