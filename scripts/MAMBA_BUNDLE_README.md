# Mamba Offline Bundle

This project uses a model path that depends on `mamba_ssm`, so we prepare an offline install bundle locally and upload it to the server.

## Local preparation

Run this on your local machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_mamba_bundle.ps1
```

That creates:

- `offline_bundle/mamba_py310_torch211_cu118/wheels/`
- `offline_bundle/mamba_py310_torch211_cu118/src/`
- `offline_bundle/mamba_py310_torch211_cu118/manifests/`

## Server install

Upload the whole `offline_bundle/mamba_py310_torch211_cu118/` directory to the repo root on the server, then run:

```bash
bash scripts/install_mamba_offline.sh
```

## Recommended server baseline

- single GPU with 40G or more is a reasonable first attempt
- prefer an `ubuntu22_cuda11.8` style image with Anaconda and full system permissions
- avoid a prebuilt Python 3.12 / Torch 2.5 image for the first run
- use Python 3.10 inside conda
- install `torch==2.1.1+cu118` inside the conda env even if the image comes with another torch version

## Notes

- The bundle is intentionally not meant to be committed to git.
- If `mamba_ssm` wheel fails, the source zips are kept as fallback.
- If you already modified `scripts/setup_server.sh` for your own environment, keep using it. This offline path is independent.
