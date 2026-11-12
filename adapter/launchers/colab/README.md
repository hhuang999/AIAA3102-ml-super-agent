# Running the Core on Google Colab

These scripts run the Core on a Colab GPU session through the `colab` command-line tool and keep the
session alive during long runs. `remote/` runs on the Colab machine, `local/` on the controlling machine.

## Files

| File | Where | Purpose |
|---|---|---|
| `launch.example.json` | — | Template of `launch.json`: paths on the Colab machine and the Core options |
| `local/pack.sh` | local | Packs the Core, the adapter and a task directory into three archives |
| `remote/setup.sh` | Colab | Creates a Python 3.11 environment, installs both projects, downloads the pretrained models, runs the Core tests |
| `remote/start_setup.py` | Colab | Starts `setup.sh` in the background |
| `remote/run.py` | Colab | Runs the Core through the adapter with the settings of `launch.json` (one task, or a batch plan given as `batch_plan`) |
| `remote/start_run.py` | Colab | Starts `run.py` in the background |
| `remote/ping.py` | Colab | Prints an `ALIVE` line and the last line of the driver log |
| `remote/snapshot.py` | Colab | Packs the output directory without model weights into `snapshot.tar.gz` and prints the run status |
| `local/keepalive.sh` | local | Contacts the session every few minutes, downloads snapshots, stops when the run ends |

## Steps

The Colab working directory is `/content`; the helper files and the run settings go to `/content/launch`.

```bash
local/pack.sh <core dir> <adapter dir> <task dir> upload/
```

```bash
colab --auth oauth2 new -s run1 --gpu L4
```

```bash
colab --auth oauth2 upload -s run1 upload/core.tar.gz upload/adapter.tar.gz upload/task.tar.gz /content/
```

Upload `remote/*`, `launch.json` (copied from `launch.example.json` and edited), the Core config file and
the `.env` file with the API keys to `/content/launch/`, then start the setup:

```bash
colab --auth oauth2 exec -s run1 -f remote/start_setup.py
```

The setup log is `/content/setup.log`; it ends with `== SETUP DONE`. Then start the run and the keep-alive loop:

```bash
colab --auth oauth2 exec -s run1 -f remote/start_run.py
```

```bash
SESSION=run1 LOCAL_DIR=snapshots local/keepalive.sh
```

When `snapshots/keepalive.log` shows `RUN FINISHED`, pack and download the full output (including the model
files if needed) and stop the session with `colab --auth oauth2 stop -s run1`.

## Notes

- A session without activity can be reclaimed. Contacting it every 180 seconds kept sessions alive
  through runs of eight hours; with intervals of 20 minutes a session was lost.
- `run.py` appends `== run end <time> exit <code>` to the driver log when the Core finishes.
- `snapshot.py` leaves out model weights, so snapshots stay small; the Core output keeps the weights.
