# COMP4405 Image Dehazing

Single-image dehazing based on the Dark Channel Prior and the atmospheric scattering model:

```
I(x) = J(x) t(x) + A (1 - t(x))
```

## Requirements

- Docker + Docker Compose

## Setup

Build the image:

```bash
docker compose build
```

## Usage

Place hazy images in `data/input/`. Dehazed results are written to `data/output/`.

Process a single image:

```bash
docker compose run --rm dehaze --input data/input/foggy.jpg --output data/output/clear.jpg
```

Process a whole directory:

```bash
docker compose run --rm dehaze --input data/input --output data/output
```

## Tunable Parameters

Defaults live in `src/constants.py` and can be overridden via CLI:

| Flag | Default | Description |
| --- | --- | --- |
| `--patch-size` | 15 | Dark-channel patch size (pixels) |
| `--omega` | 0.95 | Haze-retention factor (keeps a small amount of haze for realism) |
| `--t-min` | 0.1 | Lower bound on transmission to prevent noise amplification |
| `--atmos-top-percent` | 0.001 | Fraction of brightest dark-channel pixels used to estimate `A` |
| `--guided-radius` | 60 | Guided filter radius |
| `--guided-eps` | 1e-4 | Guided filter regularisation |
| `--gamma` | 1.0 | Post-recovery gamma correction |

Example:

```bash
docker compose run --rm dehaze \
    --input data/input/foggy.jpg \
    --output data/output/clear.jpg \
    --omega 0.9 \
    --gamma 1.2
```

## Project Structure

```
.
├── main.py                  # CLI entrypoint
├── src/
│   ├── constants.py         # Default hyperparameters
│   ├── io_utils.py          # Image I/O
│   ├── preprocessing.py     # Resize, normalise
│   ├── dark_channel.py      # Dark Channel Prior
│   ├── atmospheric_light.py # Atmospheric light A estimation
│   ├── transmission.py      # Transmission map t(x)
│   ├── refinement.py        # Guided filter refinement
│   ├── recovery.py          # Scene radiance recovery
│   ├── postprocessing.py    # Clip, convert to uint8
│   └── pipeline.py          # Orchestration
├── data/
│   ├── input/               # Hazy images go here
│   └── output/              # Dehazed results
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Running Without Docker

Make a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --input data/input --output data/output
```
At this point in time, please run with --skip-refinment and --skip-sky-detection, as the guided filter refinement and the sky detection/transmission correction is not yet implemented.
