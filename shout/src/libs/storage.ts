/**
 * Interfaces and utilities for storing and retrieving data in IndexedDB.
 * This includes the Whisper model, reconstruction lexicons, and user preferences.
 */

interface StoredData {
  model: ArrayBuffer;        // Whisper model
  lexicon: string[];         // Reconstruction lexicons
  userPreferences: {
    language: string;
    theme: 'light' | 'dark';
  };
}