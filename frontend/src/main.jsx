import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Global fetch interceptor to support remote API endpoints in production
const API_BASE_URL = import.meta.env.VITE_API_URL || '';
if (API_BASE_URL) {
  const originalFetch = window.fetch;
  window.fetch = (url, options) => {
    let targetUrl = url;
    if (typeof url === 'string' && url.startsWith('/api')) {
      const base = API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL;
      targetUrl = `${base}${url}`;
    }
    return originalFetch(targetUrl, options);
  };
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
