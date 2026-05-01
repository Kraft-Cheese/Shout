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
  // given a, b as lengths
  const m = a.length;
  const n = b.length;

  // dp is an array of size (m+1) x (n+1) where dp[i][j] is the edit distance between
  // the first i chars of a and the first j chars of b
  const dp = Array.from({ length: m + 1 }, (_, i) => [i]);

  // initialize first row and column
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  // fill in the dp table
  for (let i = 1; i <= m; i++) {

    // compute edit distance for the first i chars of a and first j chars of b
    for (let j = 1; j <= n; j++) {
      // if chars match, no edit needed; else consider insert, delete, substitute
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  // edit distance is in the bottom-right cell
  return dp[m][n];
}

/**
 * BK-tree implementation for efficient approximate string matching.
 * Each node is a word with edges to children at distances equal to their edit distance.
 * Search explores only branches that could yield a closer match than the best found so far.
 */
class BKTree {
  constructor() {
    this.root = null;
  }

  add(word) {
    if (!this.root) {
      // First word becomes root
      this.root = { word, children: {} };
      return;
    }

    // Start at root (depth first search)
    let node = this.root;
    while (true) {
      // get edit distance from current node to new word
      const d = editDistance(word, node.word);

      // If distance is 0, it's a duplicate
      if (d === 0) return; // duplicate
      // else if a child at this distance exists, go there
      if (node.children[d]) {
        // continue searching down this branch
        node = node.children[d];
      } else {
        // else create a new child at this distance
        node.children[d] = { word, children: {} };
        return;
      }
    }
  }

  // Returns the closest word within maxDist, or null if none found
  search(query, maxDist) {
    // If tree is empty, return null
    if (!this.root) return null;

    // Best match found so far
    let best = null;

    // Best distance = worse than any valid match
    let bestDist = maxDist + 1;

    // stack for DFS, starting with root
    const stack = [this.root];

    // While there are nodes in the stack
    while (stack.length) {
      // Get the next node to explore
      const node = stack.pop();

      // Compute edit distance from query to this node's word
      const d = editDistance(query, node.word);

      // If this is the best match so far, update best and bestDist
      if (d < bestDist) {
        bestDist = d;
        best = node.word;
      }

      // For each child and its distance from this node
      for (const [childDist, child] of Object.entries(node.children)) {
        const cd = parseInt(childDist);

        // If the child's distance from this node is within bestDist of the query, explore it
        if (cd >= d - maxDist && cd <= d + maxDist) {
          stack.push(child);
        }
      }
    }
    // return else null
    return bestDist <= maxDist ? best : null;
  }
}

// One BK-tree per language, built lazily on first use.
const treeCache = {};

// Build a BK-tree from the given vocabulary for efficient approximate matching.
export function buildTree(vocab) {
  const tree = new BKTree();

  // For each word in the vocabulary, add it to the BK-tree
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
  // Get the BK-tree for this language from the cache
  const tree = treeCache[langKey];

  let changed = 0;

  // Replace each word in the input text with the closest match in the BK-tree if within MAX_DIST
  const corrected = text.replace(/[\w']+/g, (word) => {
    // Normalize the word for matching (lowercase, strip punctuation)
    const lower = word.toLowerCase();

    // Search for the closest match in the BK-tree
    const match = tree.search(lower, MAX_DIST);

    // If a match is found and it's different from the original word, replace it
    if (match && match !== lower) {
      changed++;
      // Preserve original capitalisation if the input word was capitalised
      return word[0] === word[0].toUpperCase()
        ? match[0].toUpperCase() + match.slice(1)
        : match;
    }
    return word;
  });

  // Return the corrected text and the count of how many words were changed
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
 * @returns {{ text: string, tokens: Array<{text: string, p: number}>, changed: number }}
 */
export function reconstructTokens(tokens, vocab, langKey, pThreshold) {
  // If no BK-tree for this language, build and cache it
  if (!treeCache[langKey]) {
    treeCache[langKey] = buildTree(vocab);
  }

  // Get the BK-tree for this language from the cache
  const tree = treeCache[langKey];

  let changed = 0;
  const reconstructedTokens = tokens.map((token) => {
    const { text, p } = token;
    if (p >= pThreshold) return { ...token };
    const word = text.trim().replace(/[^\w']+/g, '');
    if (!word) return { ...token };
    // Search for the closest match in the BK-tree
    const match = tree.search(word.toLowerCase(), MAX_DIST);
    // If a match is found and it's different from the original word, replace it
    if (match && match !== word.toLowerCase()) {
      changed++;
      return { ...token, text: text.replace(word, match), reconstructed: true };
    }
    return { ...token, reconstructed: false };
  });

  const text = reconstructedTokens.map((t) => t.text).join('');
  return { text, tokens: reconstructedTokens, changed };
}
