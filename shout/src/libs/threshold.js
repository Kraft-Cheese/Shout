/**
 * Threshold tuning utilities for confidence-based reconstruction.
 */

import { getConfidence } from './confidence.js';

/**
 * Compute F1 score for binary predictions.
 * @param {Array<{ predicted: boolean, actual: boolean }>} predictions
 */
export function computeF1(predictions) {
  let tp = 0, fp = 0, fn = 0;

  for (const { predicted, actual } of predictions) {
    if (predicted && actual) tp++;
    else if (predicted && !actual) fp++;
    else if (!predicted && actual) fn++;
  }

  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;

  return precision + recall > 0
    ? (2 * precision * recall) / (precision + recall)
    : 0;
}

/**
 * Evaluate threshold on a development set.
 * @param {Array<{ tokens: Array, hasError: boolean }>} devSet
 * @param {number[]} thresholds
 */
export function evaluateThresholds(devSet, thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]) {
  const results = [];

  for (const theta of thresholds) {
    const predictions = devSet.map((ex) => ({
      predicted: getConfidence(ex.tokens) < theta,
      actual: ex.hasError,
    }));

    results.push({ threshold: theta, f1: computeF1(predictions) });
  }

  return results;
}