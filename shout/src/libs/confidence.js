/**
 * Confidence scoring utilities for transcription quality assessment.
 */

/**
 * Calculate confidence score from token log probabilities.
 * @param {Array<{ logprob: number }>} tokens
 * @returns {number} Confidence between 0 and 1
 */
export function getConfidence(tokens) {
  if (!tokens?.length) return 0;

  const avgLogProb = tokens.reduce((sum, t) => sum + t.logprob, 0) / tokens.length;
  return Math.exp(avgLogProb);
}

/**
 * Determine if transcription should be reconstructed based on confidence.
 * @param {number} confidence
 * @param {number} threshold
 */
export function shouldReconstruct(confidence, threshold = 0.5) {
  return confidence < threshold;
}