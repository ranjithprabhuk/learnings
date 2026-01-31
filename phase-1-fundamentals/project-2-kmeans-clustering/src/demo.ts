/**
 * K-Means Clustering Demos
 *
 * This demonstrates K-Means clustering on various datasets
 * showing different scenarios and use cases.
 */

import { KMeans } from './kmeans.js';
import { normalizeFeatures, generateBlobs, silhouetteScore } from './utils.js';

console.log('='.repeat(70));
console.log('K-MEANS CLUSTERING - DEMOS');
console.log('='.repeat(70));

// ============================================================================
// DEMO 1: Simple 2D Blobs (Perfect Separation)
// ============================================================================
console.log('\n📊 DEMO 1: Three Well-Separated Clusters (2D)');
console.log('-'.repeat(70));

const demo1Data = [
  // Cluster 1 (bottom-left)
  [1, 1], [2, 1], [1, 2], [2, 2], [1.5, 1.5],
  // Cluster 2 (top)
  [1, 9], [2, 9], [1, 10], [2, 10], [1.5, 9.5],
  // Cluster 3 (right)
  [9, 5], [10, 5], [9, 6], [10, 6], [9.5, 5.5]
];

const model1 = new KMeans(3, 100, 0.001, 'kmeans++');
model1.fit(demo1Data);

console.log('📍 Final Centroids:');
model1.getCentroids().forEach((centroid, i) => {
  console.log(`  Cluster ${i}: [${centroid.map(v => v.toFixed(2)).join(', ')}]`);
});

console.log('\n📊 Cluster Sizes:', model1.getClusterSizes());
console.log('📈 WCSS:', model1.getWCSS().toFixed(4));

const silhouette1 = silhouetteScore(demo1Data, model1.getLabels());
console.log(`🎯 Silhouette Score: ${silhouette1.toFixed(4)} (close to 1 = excellent!)`);

// ============================================================================
// DEMO 2: Customer Segmentation (Real-World Example)
// ============================================================================
console.log('\n\n📊 DEMO 2: Customer Segmentation');
console.log('-'.repeat(70));
console.log('Features: [Annual Spending ($1000s), Visit Frequency (visits/year)]');

const customerData = [
  // Budget shoppers (low spending, high frequency)
  [30, 80], [35, 75], [32, 82], [38, 78], [33, 85],
  // Occasional big spenders (high spending, low frequency)
  [180, 8], [200, 10], [195, 9], [210, 7], [190, 11],
  // Regular customers (medium spending, medium frequency)
  [90, 40], [100, 45], [95, 42], [105, 43], [98, 44],
  [85, 38], [110, 46], [92, 41]
];

// Normalize features (critical for different scales!)
const { normalized: normalizedCustomers, means, stds } = normalizeFeatures(customerData);

console.log('\n🔧 Features normalized (spending & frequency have different scales)');
console.log(`  Spending: mean=${means[0].toFixed(1)}, std=${stds[0].toFixed(1)}`);
console.log(`  Frequency: mean=${means[1].toFixed(1)}, std=${stds[1].toFixed(1)}`);

const model2 = new KMeans(3);
model2.fit(normalizedCustomers);

console.log('\n📍 Customer Segments Found:');
model2.getCentroids().forEach((centroid, i) => {
  const denormalized = [
    centroid[0] * stds[0] + means[0],
    centroid[1] * stds[1] + means[1]
  ];
  const size = model2.getClusterSizes()[i];
  console.log(`\n  Segment ${i + 1} (${size} customers):`);
  console.log(`    Avg Spending: $${denormalized[0].toFixed(0)}k/year`);
  console.log(`    Avg Visits: ${denormalized[1].toFixed(0)} times/year`);

  // Interpret segment
  if (denormalized[0] < 50 && denormalized[1] > 50) {
    console.log(`    → Budget Shoppers (frequent but low value)`);
  } else if (denormalized[0] > 150) {
    console.log(`    → VIP Customers (high value, less frequent)`);
  } else {
    console.log(`    → Regular Customers (balanced)`);
  }
});

const silhouette2 = silhouetteScore(normalizedCustomers, model2.getLabels());
console.log(`\n🎯 Silhouette Score: ${silhouette2.toFixed(4)}`);

// ============================================================================
// DEMO 3: Elbow Method (Finding Optimal K)
// ============================================================================
console.log('\n\n📊 DEMO 3: Elbow Method - Finding Optimal K');
console.log('-'.repeat(70));

// Generate synthetic data with 4 natural clusters
const elbowData = generateBlobs(4, 10, 1.5);

console.log('Testing K from 1 to 8...\n');
console.log('K  | WCSS      | Silhouette | Interpretation');
console.log('---|-----------|------------|------------------');

const wcssValues: number[] = [];

for (let k = 1; k <= 8; k++) {
  const model = new KMeans(k, 50, 0.001, 'kmeans++');
  model.fit(elbowData);

  const wcss = model.getWCSS();
  wcssValues.push(wcss);

  const silhouette = k > 1 ? silhouetteScore(elbowData, model.getLabels()) : 0;

  let interpretation = '';
  if (k === 1) interpretation = 'Too few';
  else if (k >= 2 && k <= 5) {
    if (silhouette > 0.5) interpretation = 'Good candidate';
    else interpretation = 'Consider';
  } else interpretation = 'Likely too many';

  console.log(
    `${k.toString().padStart(2)} | ${wcss.toFixed(2).padStart(9)} | ` +
    `${k > 1 ? silhouette.toFixed(4) : '   N/A'} | ${interpretation}`
  );
}

console.log('\n📊 WCSS Trend:');
for (let i = 0; i < wcssValues.length; i++) {
  const bar = '█'.repeat(Math.max(1, Math.floor(wcssValues[i] / 10)));
  console.log(`K=${i + 1}: ${bar} (${wcssValues[i].toFixed(1)})`);
}

console.log('\n💡 Look for the "elbow" where WCSS decrease slows down');
console.log('💡 Higher silhouette score = better clustering');

// Calculate improvement rate
console.log('\n📉 WCSS Improvement Rate:');
for (let i = 1; i < wcssValues.length; i++) {
  const improvement = ((wcssValues[i - 1] - wcssValues[i]) / wcssValues[i - 1]) * 100;
  const marker = improvement < 15 ? ' ← Elbow here?' : '';
  console.log(`K=${i} → K=${i + 1}: ${improvement.toFixed(1)}% improvement${marker}`);
}

// ============================================================================
// DEMO 4: Impact of Feature Scaling
// ============================================================================
console.log('\n\n📊 DEMO 4: Why Feature Scaling Matters');
console.log('-'.repeat(70));

const scalingData = [
  [1000, 2], [1100, 3], [1050, 2],    // Cluster 1
  [3000, 8], [3100, 9], [3050, 8],    // Cluster 2
];

console.log('\n❌ WITHOUT Normalization:');
console.log('Feature 1 (1000s) dominates Feature 2 (single digits)\n');

const model4a = new KMeans(2);
model4a.fit(scalingData);

console.log('Centroids:');
model4a.getCentroids().forEach((c, i) => {
  console.log(`  Cluster ${i}: [${c[0].toFixed(1)}, ${c[1].toFixed(1)}]`);
});
console.log(`WCSS: ${model4a.getWCSS().toFixed(2)}`);

console.log('\n✅ WITH Normalization:');
console.log('Both features equally important\n');

const { normalized: scalingNormalized } = normalizeFeatures(scalingData);
const model4b = new KMeans(2);
model4b.fit(scalingNormalized);

console.log('Centroids (normalized):');
model4b.getCentroids().forEach((c, i) => {
  console.log(`  Cluster ${i}: [${c[0].toFixed(4)}, ${c[1].toFixed(4)}]`);
});
console.log(`WCSS: ${model4b.getWCSS().toFixed(4)}`);

const silhouette4a = silhouetteScore(scalingData, model4a.getLabels());
const silhouette4b = silhouetteScore(scalingNormalized, model4b.getLabels());
console.log(`\nSilhouette without normalization: ${silhouette4a.toFixed(4)}`);
console.log(`Silhouette with normalization:    ${silhouette4b.toFixed(4)} ← Better!`);

// ============================================================================
// DEMO 5: K-Means++ vs Random Initialization
// ============================================================================
console.log('\n\n📊 DEMO 5: Initialization Methods Comparison');
console.log('-'.repeat(70));

const initData = generateBlobs(3, 15, 2);

console.log('Running both methods 5 times each...\n');

let randomWcssAvg = 0;
let kmeansppWcssAvg = 0;

// Random initialization
console.log('Random Initialization:');
for (let i = 0; i < 5; i++) {
  const model = new KMeans(3, 50, 0.001, 'random');
  model.fit(initData);
  const wcss = model.getWCSS();
  randomWcssAvg += wcss;
  console.log(`  Run ${i + 1}: WCSS = ${wcss.toFixed(2)}`);
}
randomWcssAvg /= 5;

// K-Means++
console.log('\nK-Means++ Initialization:');
for (let i = 0; i < 5; i++) {
  const model = new KMeans(3, 50, 0.001, 'kmeans++');
  model.fit(initData);
  const wcss = model.getWCSS();
  kmeansppWcssAvg += wcss;
  console.log(`  Run ${i + 1}: WCSS = ${wcss.toFixed(2)}`);
}
kmeansppWcssAvg /= 5;

console.log('\n📊 Results:');
console.log(`  Random init average WCSS:    ${randomWcssAvg.toFixed(2)}`);
console.log(`  K-Means++ average WCSS:      ${kmeansppWcssAvg.toFixed(2)} ← More consistent!`);
console.log(`  Improvement:                 ${((randomWcssAvg - kmeansppWcssAvg) / randomWcssAvg * 100).toFixed(1)}%`);

// ============================================================================
// KEY LEARNINGS
// ============================================================================
console.log('\n\n' + '='.repeat(70));
console.log('📚 KEY LEARNINGS');
console.log('='.repeat(70));
console.log(`
1. **K-Means is Unsupervised Learning**
   - No labels needed!
   - Finds natural groupings in data
   - Different from linear regression (supervised)

2. **Feature Scaling is CRITICAL** ⭐
   - Features with large values dominate distance calculations
   - Always normalize before clustering
   - Demo 4 shows dramatic difference

3. **Choosing K**
   - Use elbow method (look for "bend" in WCSS curve)
   - Use silhouette score (>0.5 is good)
   - Consider domain knowledge
   - Demo 3 shows how to find optimal K

4. **Initialization Matters**
   - K-Means++ gives more consistent results
   - Random init can get stuck in poor local optima
   - Always run multiple times or use K-Means++

5. **Evaluation Metrics**
   - WCSS (lower is better, but always decreases with more K)
   - Silhouette score (0.7-1.0 = strong, 0.5-0.7 = reasonable)
   - Visual inspection (for 2D/3D data)

6. **Real-World Applications**
   - Customer segmentation (Demo 2)
   - Image compression
   - Document clustering
   - Anomaly detection
   - Feature preprocessing

7. **Limitations**
   - Assumes spherical clusters
   - Sensitive to outliers
   - Must choose K beforehand
   - Can get stuck in local optima

8. **Best Practices**
   - Always normalize features
   - Try multiple K values
   - Use K-Means++ initialization
   - Run multiple times, keep best result
   - Visualize when possible
   - Validate clusters make business sense
`);

console.log('='.repeat(70));
console.log('✅ Demo Complete! Try modifying K values and datasets.');
console.log('='.repeat(70) + '\n');
