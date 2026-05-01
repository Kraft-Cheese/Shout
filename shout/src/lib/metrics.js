/**
 * Metrics for ASR evaluation.
 * WER, CER, F1 (morpheme and boundary).
 */

/**
 * Text normalisation
 * Lowercase, strip punctuation (keep letters, digits, spaces, hyphens), collapse whitespace.
 */
function normalise(text) {
  if (!text) return "";
  // Lowercase
  let t = text.toLowerCase();
  // Strip punctuation (keep letters, digits, spaces, hyphens)
  // Python: re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
  // In JS, \w is [A-Za-z0-9_]. Python's \w with UNICODE includes all Unicode letters.
  // We should use a regex that supports Unicode letters if possible, or at least match the Python intent.
  t = t.replace(/[^\p{L}\p{N}\s\-]/gu, " ");
  // Collapse whitespace and trim
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

/**
 * Word Error Rate (WER)
 */
export function calculateWER(reference, hypothesis) {
  const refWords = normalise(reference).split(/\s+/).filter(w => w.length > 0);
  const hypWords = normalise(hypothesis).split(/\s+/).filter(w => w.length > 0);

  if (refWords.length === 0) return hypWords.length === 0 ? 0 : Infinity;

  const d = levenshteinMatrix(refWords, hypWords);
  return d[refWords.length][hypWords.length] / refWords.length;
}

/**
 * Character Error Rate (CER)
 */
export function calculateCER(reference, hypothesis) {
  const refChars = normalise(reference).split('');
  const hypChars = normalise(hypothesis).split('');

  if (refChars.length === 0) return hypChars.length === 0 ? 0 : Infinity;

  const d = levenshteinMatrix(refChars, hypChars);
  return d[refChars.length][hypChars.length] / refChars.length;
}

/**
 * Levenshtein Distance Matrix
 */
function levenshteinMatrix(s1, s2) {
  const m = s1.length;
  const n = s2.length;
  const d = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) d[i][0] = i;
  for (let j = 0; j <= n; j++) d[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
      d[i][j] = Math.min(
        d[i - 1][j] + 1,      // deletion
        d[i][j - 1] + 1,      // insertion
        d[i - 1][j - 1] + cost // substitution
      );
    }
  }
  return d;
}

/**
 * F1 Morpheme and Boundary
 */
export function calculateF1(reference, hypothesis) {
  const morphemeF1 = tokenF1(reference, hypothesis);
  const boundaryF1 = computeBoundaryF1(reference, hypothesis);
  
  return { morphemeF1, boundaryF1 };
}

/**
 * Token-level F1 treating the sentence as a bag of words. (Morpheme F1 proxy)
 */
function tokenF1(ref, pred) {
  const refNorm = normalise(ref);
  const predNorm = normalise(pred);
  
  const refTokens = refNorm.split(" ").filter(t => t.length > 0);
  const predTokens = predNorm.split(" ").filter(t => t.length > 0);
  
  if (refTokens.length === 0 || predTokens.length === 0) {
    return 0.0;
  }
  
  const refCounts = {};
  for (const t of refTokens) refCounts[t] = (refCounts[t] || 0) + 1;
  
  const predCounts = {};
  for (const t of predTokens) predCounts[t] = (predCounts[t] || 0) + 1;
  
  let tp = 0;
  for (const t in refCounts) {
    if (predCounts[t]) {
      tp += Math.min(refCounts[t], predCounts[t]);
    }
  }
  
  const precision = tp / predTokens.length;
  const recall = tp / refTokens.length;
  
  if (precision + recall === 0) return 0.0;
  return (2 * precision * recall) / (precision + recall);
}

/**
 * Boundary positions in normalised text
 */
function getBoundaryPositions(text) {
  const norm = normalise(text);
  const positions = new Set();
  for (let i = 0; i < norm.length; i++) {
    if (norm[i] === " ") {
      positions.add(i);
    }
  }
  return positions;
}

/**
 * Word-boundary F1 (character-aligned boundary detection)
 */
function computeBoundaryF1(ref, pred) {
  const refB = getBoundaryPositions(ref);
  const predB = getBoundaryPositions(pred);
  
  if (refB.size === 0 && predB.size === 0) return 1.0;
  if (refB.size === 0 || predB.size === 0) return 0.0;
  
  let tp = 0;
  for (const pos of predB) {
    if (refB.has(pos)) tp++;
  }
  
  const precision = tp / predB.size;
  const recall = tp / refB.size;
  
  if (precision + recall === 0) return 0.0;
  return (2 * precision * recall) / (precision + recall);
}
