import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { Plugin } from 'vite';
import { defineConfig } from 'vitest/config';

/** Dev-only: accept PNG captures from the running app and store them under artifacts/captures. */
function captureEndpoint(): Plugin {
  return {
    name: 'capture-endpoint',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__capture', (req, res) => {
        const url = new URL(req.url ?? '', 'http://localhost');
        const name = (url.searchParams.get('name') ?? 'capture').replace(/[^\w.-]/g, '_');
        const chunks: Buffer[] = [];
        req.on('data', (c: Buffer) => chunks.push(c));
        req.on('end', () => {
          const dir = resolve(import.meta.dirname, 'artifacts/captures');
          mkdirSync(dir, { recursive: true });
          writeFileSync(resolve(dir, `${name}.png`), Buffer.concat(chunks));
          res.statusCode = 204;
          res.end();
        });
      });
    },
  };
}

export default defineConfig({
  server: {
      allowedHosts: true
  },
  plugins: [captureEndpoint()],
  worker: { format: 'es' },
  build: { target: 'es2023' },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
