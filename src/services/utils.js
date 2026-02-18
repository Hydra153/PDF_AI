export function sanitizeInput(text = '') {
  const blocked = [/ignore previous instructions/i, /forget everything/i];
  let sanitized = text;
  for (const pattern of blocked) {
    sanitized = sanitized.replace(pattern, '[redacted]');
  }
  return sanitized;
}

export function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('Timeout')), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
