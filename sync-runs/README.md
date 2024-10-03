# Upload local runs to some remote MLFLow server

As a first step, build the environment using `requirements.txt` and launch training:

```bash
python train_local.py
```

This scripts will create MLFlow logs locally under `./mlruns` directory.

Second, identify the run ID of the run to upload and export it:

```bash
export RUNID="01477cbaa3774ca998eb5f4278aea892"
```

Now, there are two ways to upload local runs to the remote MLFlow server:

1. Direct copy: run `copy.sh`. **This is the suggested way to go.**
2. Export to some local folder running `export.sh` and upload the exported data to the server running `import.sh`

The bash scripts provided are just a minimal example and *should* be adapted to
the user needs. However, they provide a minimal working example as a starting point.

> [!WARNING]
> This can be done only once the MLFlow run (i.e., ML training) is finished!
