# Scripts

Script groups:

- `dehaze.sh`: small daily-use wrapper for image, video, evaluation, and training.
- `data/`: download and normalize local datasets.
- `eval/`: run repeatable benchmark iterations from a manifest.
- `neural/`: train and run the RGB-only DCP-inspired PyTorch dehazer.

All paths are resolved from the repository root, so scripts can be run from any
current working directory.

Common use:

```bash
scripts/dehaze.sh image hazy.jpg out.png
scripts/dehaze.sh video input.mp4 out.mp4
scripts/dehaze.sh eval
METHOD=improved-dcp BACKEND=cpu scripts/dehaze.sh image hazy.jpg out_dcp.png
```
