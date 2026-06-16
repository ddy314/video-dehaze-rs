# Crates

- `dehaze-core`: original DCP, improved traditional DCP, temporal stabilization, synthetic haze, and metrics.
- `dehaze-cli`: command-line dispatch plus runtime IO for images, videos, evaluation, and Python neural inference.

Common commands:

```bash
cargo run -p dehaze-cli -- image input.png -o output.png --method improved-dcp
cargo run -p dehaze-cli -- video input.mp4 -o output.mp4 --method improved-dcp --simd auto
cargo run -p dehaze-cli -- image input.png -o output.png --method neural --backend gpu --model models/neural_dehazer.pt
```
