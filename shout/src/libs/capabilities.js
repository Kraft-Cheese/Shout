/**
 * Browser capability detection.
 * Checks for features required by Whisper ASR.
 */

const MIN_STORAGE_MB = 100;

/**
 * Check browser capabilities and return any issues found.
 * @returns {Promise<{ supported: boolean, issues: string[], recommendations: string[] }>}
 */
export async function checkCapabilities() {
  const issues = [];

  // WebAssembly support
  if (typeof WebAssembly === 'undefined') {
    issues.push('WebAssembly not supported');
  }

  // SharedArrayBuffer for threading
  if (typeof SharedArrayBuffer === 'undefined') {
    issues.push('SharedArrayBuffer not available (threading disabled)');
  }

  // Microphone access
  if (!navigator.mediaDevices?.getUserMedia) {
    issues.push('Microphone access not supported');
  }

  // Storage availability
  if (navigator.storage?.estimate) {
    const { quota = 0, usage = 0 } = await navigator.storage.estimate();
    const availableMB = (quota - usage) / (1024 * 1024);
    if (availableMB < MIN_STORAGE_MB) {
      issues.push('Low storage space - model caching may fail');
    }
  }

  return {
    supported: issues.length === 0,
    issues,
    recommendations: getRecommendations(issues),
  };
}

/**
 * Generate user-friendly recommendations based on detected issues.
 */
function getRecommendations(issues) {
  const recommendations = [];

  if (issues.some((i) => i.includes('SharedArrayBuffer'))) {
    recommendations.push('Use Chrome or Firefox for best performance');
    recommendations.push('Processing will be slower without threading');
  }

  if (issues.some((i) => i.includes('storage'))) {
    recommendations.push('Clear browser cache to free space');
  }

  return recommendations;
}