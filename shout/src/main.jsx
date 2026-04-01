import { render } from 'preact';
import './index.css';
import { App } from './app.jsx';

render(<App />, document.getElementById('app'));

// Register service worker for offline support
// COOP/COEP headers directly via vite.config.js.
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register('/sw.js', { scope: '/' });
}
