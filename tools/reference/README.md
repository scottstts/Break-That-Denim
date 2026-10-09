# Reference-frame extraction

From the project root:

```sh
zsh -ic 'source tools/reference/extract-frames.zsh'
```

To choose an input or a new output directory:

```sh
zsh -ic 'source tools/reference/extract-frames.zsh refs/demo.mov artifacts/reference/second-study'
```

Requires `ffmpeg`, `ffprobe`, and `shasum` available in the interactive shell. No package installation is performed. Existing output directories are rejected rather than overwritten. A failed extraction leaves its logs available for diagnosis; retry with a new directory after addressing the error.

The tool writes eight native-resolution frames, an unlabelled contact sheet in chronological order, source metadata and SHA-256, the FFmpeg version, and per-frame decode logs. Filenames record the requested timestamp; `showinfo` in the logs records the actual selected presentation timestamp. This avoids equating nominal frame number with elapsed time.

Generated output defaults to `artifacts/reference/study/` and is ignored by Git. Preserve the original video and contact sheet in `refs/`. For observations and comparison guidance, see [reference analysis](../../docs/research/reference-analysis.md).
