// Headless GPU runner: executes a TypeScript entry (with Vite transforms such as `?raw` WGSL imports)
// against Dawn's Metal backend in Node. No browser is involved.
//   node tools/gpu/run.mjs tools/gpu/scripts/render.ts [args...]
import { create, globals } from 'webgpu';
import { resolve } from 'node:path';
import { createServer, createServerModuleRunner } from 'vite';

Object.assign(globalThis, globals);
const gpu = create([]);
Object.defineProperty(globalThis.navigator, 'gpu', { value: gpu, configurable: true });

const root = resolve(import.meta.dirname, '../..');
const server = await createServer({
  root,
  configFile: resolve(root, 'vite.config.ts'),
  logLevel: 'error',
  server: { middlewareMode: true, hmr: false, watch: null },
  appType: 'custom',
});
const runner = createServerModuleRunner(server.environments.ssr, { hmr: false });
let code = 0;
try {
  const entry = resolve(process.cwd(), process.argv[2]);
  const mod = await runner.import(entry);
  await mod.default(process.argv.slice(3));
} catch (e) {
  console.error(e);
  code = 1;
} finally {
  await runner.close();
  await server.close();
}
process.exit(code);
