// Confidence level mapping for styling
const confidenceLevel = (p) => (p > 0.7 ? 'high' : p > 0.4 ? 'medium' : 'low');

/**
 * Build an interpunct display model from token data.
 * The caller can render these parts as spans while keeping the logic here.
 * @param {Array<{text: string, p: number, t0: number}>} tokens
 * @returns {Array<{kind: 'token' | 'separator', key: string, className: string, title?: string, text: string}>}
 */
export function buildInterpunctParts(tokens) {
  return tokens.flatMap((token, index) => {
    const parts = [
      {
        kind: 'token',
        key: `${index}-token`,
        className: `token confidence-${confidenceLevel(token.p)}`,
        title: `p=${(token.p * 100).toFixed(1)}%  t0=${token.t0 * 10}ms`,
        text: token.text,
      },
    ];

    if (index < tokens.length - 1) {
      parts.push({
        kind: 'separator',
        key: `${index}-dot`,
        className: 'interpunct',
        text: '·',
      });
    }

    return parts;
  });
}
