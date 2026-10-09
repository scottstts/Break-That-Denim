import { defineConfig } from 'vitest/config';

export default defineConfig({
  worker: { format: 'es' },
  build: { target: 'es2023' },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
