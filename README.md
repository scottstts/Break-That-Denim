# Break That Denim

![Denim, mid-tear](assets/screenshot.jpeg)

**Grab it. Pull. Listen to it give.**

Everyone knows the feeling: the first tiny rip in a pair of jeans, the way it hesitates, then *runs*. The pale threads that stretch across the gap like they're begging you to stop. The frayed edge that looks like it was drawn by someone who really committed to the bit.

This is that moment, in your browser, for as long as you want it.

**Play it: [denim.scottsun.io](https://denim.scottsun.io)** (needs a browser with WebGPU)

---

## It isn't an animation. Nothing is scripted.

There is no "tear" clip. No alpha mask eating the cloth away. No grid of springs that snap at a threshold and call it fabric.

Under your fingers is a piece of denim woven out of individual yarns: blue warp, pale weft, the real construction. Pull it and the load travels along the threads that are actually holding. The weak, worn ones let go first. The tear finds its own path. Threads that survive across the opening keep carrying force, so the cloth on either side feels them, right up until the moment they pop. Cut a thread and the physics of everything around it changes.

A hole here is *absent material*, not a texture painted black. It's missing from the mechanics, the collisions, the shadows and the picking alike.

Let go mid-rip and the stored elastic energy can finish the job on its own. Sometimes it stops. Sometimes it doesn't. That's the fun.

## The details that sell it

- **Yarns, not triangles.** Where the cloth is about to fail, it stops being a smooth sheet and becomes individual woven yarns that can slip, pull out and snap.
- **Fuzz with a family tree.** The frayed edge is grown from the fibres of the yarn that broke, so it looks the way real frayed denim looks.
- **Hanging threads that behave.** Loose strands dangle, swing and tangle like the real thing.
- **Pinch with your hand.** Grab, twist, drag, with multitouch. Pieces you tear free are yours: they fall, tumble and land.
- **A sound that is only the tearing.** The audio is synthesised live from what the cloth is doing. It isn't a sample on a loop. No rip, no sound; a harder rip is louder and sharper.
- **Soft, believable light.** A room, a floor, directional shadows, and a weave you can see.

## Under the hood

Everything runs locally, on your GPU. No server, no pre-baked physics.

| | |
|---|---|
| **Solver** | A coupled implicit solve on WebGPU (WGSL compute): multigrid-preconditioned PCG, contact, self-contact, grips and yarn mechanics in one system |
| **Fabric** | Anisotropic thin shell for intact cloth, explicit woven yarns where damage happens, elastic rods for loose strands |
| **Renderer** | Custom WebGPU, with yarn and fibre reconstruction filtered to the pixel footprint |
| **Audio** | Procedural tear voice driven by the simulation (on/off, loudness, stereo pan) |
| **Budget** | Hard 2M-pixel drawing-buffer cap and a real-time frame budget. Realism isn't allowed to cost smoothness |

The guiding rule: **realism comes from getting the mechanism right**, not from faking the look. If a shortcut would make a screenshot prettier by breaking the engine, it doesn't ship.

## And then there's the stage

Beside the Reset button there are other ways to be handed the cloth. They're staged, and they're a little unsettling. The tearing is still the real thing, driven by the same mechanics. Poke around if you dare.

## Run it

```bash
npm install
npm run dev
```

Use a recent desktop Chrome. Apple silicon Macs are the primary target.

```bash
npm run check      # lint, typecheck, tests
npm run test:gpu   # headless GPU correctness suites
npm run build      # production bundle
```

## Poke at the thinking

The design notes, research and every decision made along the way live in [`docs/`](docs): [`design.md`](docs/design.md) is the place to start.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0-only).

You may use, modify, and redistribute this software under the terms of the GPLv3. If you distribute a modified or derivative version of this project, you must also make the corresponding source code available under the GPLv3.

See the [LICENSE](./LICENSE) file for the full license terms.

---

*Made for people who have never been able to resist a loose thread.*