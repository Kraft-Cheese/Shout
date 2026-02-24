/**
 * Device detection and capability assessment.
 */

const MOBILE_REGEX = /Android|iPhone|iPad|iPod/i;

/**
 * Check if the current device is mobile.
 */
export function isMobile() {
  return MOBILE_REGEX.test(navigator.userAgent);
}

/**
 * Select appropriate model size based on device capabilities.
 * @returns {'tiny' | 'base' | 'small'}
 */
export function getRecommendedModel() {
  if (isMobile()) return 'tiny';

  const memoryGB = navigator.deviceMemory || 4;

  if (memoryGB <= 4) return 'tiny';
  if (memoryGB <= 8) return 'base';
  return 'small';
}

/**
 * Get maximum recording duration in seconds based on device.
 */
export function getMaxRecordingDuration() {
  return isMobile() ? 30 : 60;
}