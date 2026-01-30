/**
 * Demo: Linear Regression
 *
 * This demonstrates the linear regression implementation
 * with various datasets and learning scenarios.
 */

import { LinearRegression } from './linear-regression.js';

console.log('='.repeat(70));
console.log('LINEAR REGRESSION FROM SCRATCH - DEMO');
console.log('='.repeat(70));

// ============================================================================
// DEMO 1: Simple Linear Relationship (y = 2x + 3)
// ============================================================================
console.log('\n📊 DEMO 1: Perfect Linear Data (y = 2x + 3)');
console.log('-'.repeat(70));

const X1 = [[1], [2], [3], [4], [5]];
const y1 = [5, 7, 9, 11, 13];

const model1 = new LinearRegression(0.01, 1000);
model1.fit(X1, y1);

// Make predictions
const predictions1 = model1.predict([[6], [7], [10]]);
console.log('\n🔮 Predictions:');
console.log(`x = 6  → y = ${predictions1[0].toFixed(2)} (expected: 15)`);
console.log(`x = 7  → y = ${predictions1[1].toFixed(2)} (expected: 17)`);
console.log(`x = 10 → y = ${predictions1[2].toFixed(2)} (expected: 23)`);

const r2_score1 = model1.score(X1, y1);
console.log(`\n📈 R² Score: ${r2_score1.toFixed(6)} (1.0 = perfect fit)`);

// ============================================================================
// DEMO 2: House Price Prediction (WITH FEATURE NORMALIZATION)
// ============================================================================
console.log('\n\n📊 DEMO 2: House Price Prediction (Size → Price) - WITH NORMALIZATION');
console.log('-'.repeat(70));

// Size in sqft → Price in $1000s
const X2 = [
  [1000],
  [1500],
  [2000],
  [2500],
  [3000],
  [3500],
];
const y2 = [200, 280, 350, 420, 500, 580];

// ⭐ KEY FIX: Enable normalization for large numbers!
const model2 = new LinearRegression(0.01, 2000, true);  // normalize=true
model2.fit(X2, y2);

// Predict prices for new houses
const predictions2 = model2.predict([[1750], [2750], [4000]]);
console.log('\n🔮 Predictions:');
console.log(`1750 sqft → $${(predictions2[0] * 1000).toFixed(0)}`);
console.log(`2750 sqft → $${(predictions2[1] * 1000).toFixed(0)}`);
console.log(`4000 sqft → $${(predictions2[2] * 1000).toFixed(0)}`);

const r2_score2 = model2.score(X2, y2);
console.log(`\n📈 R² Score: ${r2_score2.toFixed(6)}`);

// ============================================================================
// DEMO 3: Data with Noise
// ============================================================================
console.log('\n\n📊 DEMO 3: Noisy Data (Real-world scenario)');
console.log('-'.repeat(70));

// y = 3x + 2 + noise
const X3 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const y3 = [5.1, 7.8, 11.2, 13.9, 17.2, 19.8, 23.1, 25.9, 29.2, 31.8];

const model3 = new LinearRegression(0.01, 1500);
model3.fit(X3, y3);

const r2_score3 = model3.score(X3, y3);
console.log(`\n📈 R² Score: ${r2_score3.toFixed(6)}`);
console.log('Note: Lower R² due to noise in data (realistic scenario)');

// ============================================================================
// DEMO 4: Multivariate Linear Regression (WITH NORMALIZATION)
// ============================================================================
console.log('\n\n📊 DEMO 4: Multivariate Regression (Size + Bedrooms → Price) - WITH NORMALIZATION');
console.log('-'.repeat(70));

// [size, bedrooms] → price
const X4 = [
  [1000, 2],
  [1500, 3],
  [2000, 3],
  [2500, 4],
  [3000, 4],
  [3500, 5],
];
const y4 = [200, 300, 380, 480, 550, 650];

// ⭐ KEY FIX: Enable normalization for multiple features with different scales!
const model4 = new LinearRegression(0.01, 2000, true);  // normalize=true
model4.fit(X4, y4);

// Predict for new houses
const predictions4 = model4.predict([
  [1800, 3],
  [2800, 4],
]);
console.log('\n🔮 Predictions:');
console.log(`1800 sqft, 3 bedrooms → $${(predictions4[0] * 1000).toFixed(0)}`);
console.log(`2800 sqft, 4 bedrooms → $${(predictions4[1] * 1000).toFixed(0)}`);

const r2_score4 = model4.score(X4, y4);
console.log(`\n📈 R² Score: ${r2_score4.toFixed(6)}`);

// ============================================================================
// DEMO 5: Feature Scaling Comparison (Educational)
// ============================================================================
console.log('\n\n📊 DEMO 5: Why Feature Normalization Matters');
console.log('-'.repeat(70));

const X5_demo = [[2000]];
const y5_demo = [400];

console.log('\n❌ WITHOUT Normalization (Large numbers):');
const model5a = new LinearRegression(0.00001, 500, false);
model5a.fit(X5_demo, y5_demo);
const r2_5a = model5a.score(X5_demo, y5_demo);
console.log(`R² Score: ${r2_5a.toFixed(6)}`);
console.log('Problem: Cost explodes to Infinity/NaN!');

console.log('\n✅ WITH Normalization (Same data):');
const model5b = new LinearRegression(0.01, 500, true);
model5b.fit(X5_demo, y5_demo);
const r2_5b = model5b.score(X5_demo, y5_demo);
console.log(`R² Score: ${r2_5b.toFixed(6)}`);
console.log('Solution: Features scaled to similar ranges, gradient descent converges!');

// ============================================================================
// DEMO 6: Comparing Different Learning Rates
// ============================================================================
console.log('\n\n📊 DEMO 6: Impact of Learning Rate');
console.log('-'.repeat(70));

const X6 = [[1], [2], [3], [4], [5]];
const y6 = [2, 4, 6, 8, 10];

console.log('\nLearning Rate = 0.1 (Good):');
const model6a = new LinearRegression(0.1, 100);
model6a.fit(X6, y6);
console.log(`Final R² Score: ${model6a.score(X6, y6).toFixed(6)}`);

console.log('\nLearning Rate = 0.001 (Too small - slow):');
const model6b = new LinearRegression(0.001, 100);
model6b.fit(X6, y6);
console.log(`Final R² Score: ${model6b.score(X6, y6).toFixed(6)}`);
console.log('Note: Needs more iterations to converge!');

// ============================================================================
// Summary & Key Learnings
// ============================================================================
console.log('\n\n' + '='.repeat(70));
console.log('📚 KEY LEARNINGS');
console.log('='.repeat(70));
console.log(`
1. **Gradient Descent Works!**
   - Cost decreases with each iteration
   - Model learns optimal parameters (theta)

2. **Feature Normalization is CRITICAL!** ⭐
   - Large numbers (1000s) → NaN/Infinity without normalization
   - Solution: Scale features to similar ranges (z-score normalization)
   - Formula: x_norm = (x - mean) / std_dev
   - Allows using reasonable learning rates (0.01 instead of 0.00001)
   - Demo 5 shows dramatic difference!

3. **Learning Rate Matters**
   - Too high: May not converge (cost increases)
   - Too low: Slow convergence (many iterations needed)
   - Sweet spot: 0.01-0.1 (WITH normalization)
   - Without normalization: Need tiny rates (0.00001) → often fails

4. **R² Score Interpretation**
   - 1.0 = Perfect fit
   - > 0.9 = Excellent fit
   - 0.7-0.9 = Good fit
   - < 0.7 = Poor fit (may need more features or different model)

5. **Real Data Has Noise**
   - Perfect fit (R² = 1.0) only happens with synthetic data
   - Real-world data: R² of 0.7-0.9 is good!

6. **Multiple Features**
   - Can include multiple input features
   - Each feature gets its own parameter (theta)
   - ALWAYS normalize when features have different scales!
   - More features ≠ always better (risk of overfitting)
`);

console.log('='.repeat(70));
console.log('✅ Demo Complete! Try modifying the datasets and parameters.');
console.log('='.repeat(70) + '\n');
