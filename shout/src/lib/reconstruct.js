/**
 * Lexicon-based reconstruction using a BK-tree.
 *
 * Port of reconstruct_b4.py replaces words in ASR output with the closest
 * match in a known vocabulary when edit distance <= MAX_DIST.
 *
 * Used as the reconstruction stage in the Shout pipeline:
 *   ASR output -> reconstruct(text, vocab) -> corrected text
 *
 * Only fires when the worker's confidence is below threshold Tau, or can be
 * applied token-by-token targeting only low-p tokens.
 */

const MAX_DIST = 2;

function editDistance(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) => [i]);
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}

class BKTree {
  constructor() {
    this.root = null;
  }

  add(word) {
    if (!this.root) {
      this.root = { word, children: {} };
      return;
    }
    let node = this.root;
    while (true) {
      const d = editDistance(word, node.word);
      if (d === 0) return; // duplicate
      if (node.children[d]) {
        node = node.children[d];
      } else {
        node.children[d] = { word, children: {} };
        return;
      }
    }
  }

  // Returns the closest word within maxDist, or null if none found
  search(query, maxDist) {
    if (!this.root) return null;
    let best = null;
    let bestDist = maxDist + 1;
    const stack = [this.root];
    while (stack.length) {
      const node = stack.pop();
      const d = editDistance(query, node.word);
      if (d < bestDist) {
        bestDist = d;
        best = node.word;
      }
      for (const [childDist, child] of Object.entries(node.children)) {
        const cd = parseInt(childDist);
        if (cd >= d - maxDist && cd <= d + maxDist) {
          stack.push(child);
        }
      }
    }
    return bestDist <= maxDist ? best : null;
  }
}

// One BK-tree per language, built lazily on first use.
const treeCache = {};

export function buildTree(vocab) {
  const tree = new BKTree();
  for (const word of vocab) tree.add(word);
  return tree;
}

/**
 * Reconstruct ASR output against a vocabulary.
 *
 * @param {string} text - Raw ASR output
 * @param {string[]} vocab - Known wordforms for the target language
 * @param {string} langKey - Cache key (e.g. 'sei', 'ncx')
 * @returns {{ text: string, changed: number }} - Corrected text and count of substitutions
 */
export function reconstruct(text, vocab, langKey) {
  if (!treeCache[langKey]) {
    treeCache[langKey] = buildTree(vocab);
  }
  const tree = treeCache[langKey];

  let changed = 0;
  const corrected = text.replace(/[\w']+/g, (word) => {
    const lower = word.toLowerCase();
    const match = tree.search(lower, MAX_DIST);
    if (match && match !== lower) {
      changed++;
      // Preserve original capitalisation if the input word was capitalised
      return word[0] === word[0].toUpperCase()
        ? match[0].toUpperCase() + match.slice(1)
        : match;
    }
    return word;
  });

  return { text: corrected, changed };
}

/**
 * Only correct tokens below a confidence threshold.
 * avoids corrupting confident tokens.
 *
 * @param {Array<{text: string, p: number}>} tokens
 * @param {string[]} vocab
 * @param {string} langKey
 * @param {number} pThreshold - Tokens with p below this are candidates
 * @returns {{ text: string, changed: number }}
 */
export function reconstructTokens(tokens, vocab, langKey, pThreshold) {
  if (!treeCache[langKey]) {
    treeCache[langKey] = buildTree(vocab);
  }
  const tree = treeCache[langKey];

  let changed = 0;
  const parts = tokens.map(({ text, p }) => {
    if (p >= pThreshold) return text;
    const word = text.trim().replace(/[^\w']+/g, '');
    if (!word) return text;
    const match = tree.search(word.toLowerCase(), MAX_DIST);
    if (match && match !== word.toLowerCase()) {
      changed++;
      return text.replace(word, match);
    }
    return text;
  });

  return { text: parts.join(''), changed };
}
