# DGX raw SynthHuman access

The active DGX loader reads original RGB PNG, alpha PNG, depth EXR, and normal EXR files from the existing Drive folder. It does not create TAR shards. The audit uses `rclone lsf` to list filenames and saves a local deterministic manifest; the training loader opens and preprocesses the selected files on demand.

## Mount Drive with a bounded cache on `/raid`

The first mount command used `--vfs-cache-mode full` without choosing the cache directory or limiting cache size. For raw training, place the cache on `/raid`, where there is more free capacity than the system disk. Stop the existing foreground mount with **Ctrl+C** in the terminal running it, then remount:

```bash
mkdir -p "$HOME/raid/Clara/relight_runtime/rclone_vfs"
rclone mount xxdrive: "$HOME/xxdrive" \
  --vfs-cache-mode full \
  --cache-dir "$HOME/raid/Clara/relight_runtime/rclone_vfs" \
  --vfs-cache-max-size 256G \
  --vfs-cache-max-age 72h \
  --dir-cache-time 1h
```

Keep that terminal running. In a second SSH session verify the source folder:

```bash
ls "$HOME/xxdrive/AppleCapstoneRelighting/dataset/sources/SynthHuman"
```

The exact paths can be changed in `configs/dgx_fastvit_t8_depth_normals_50k.yaml` if the mount location differs.

## Audit then train

```bash
./scripts/audit_dgx.sh
```

The audit recursively lists remote paths and Google Drive object IDs through the `xxdrive:` rclone remote, groups files by numeric sample ID, excludes incomplete or ambiguous groups (including duplicate Drive filenames), and selects the requested number of complete samples with seed 42. It decodes one sample to verify RGB, EXR depth, EXR normals, alpha mask, units and tensor shapes. The inventory is cached under `/raid/Clara/relight_runtime/synthhuman_file_inventory_v1`; each run stores its exact sample/split manifest, so the pilot and full run do not repeat the remote metadata scan.

```bash
./scripts/train_dgx.sh
```

The loader reads four source files for each sample. H100 compute does not remove Drive/network latency, so monitor `data_s` in the per-batch progress and `GPU-Util` in `nvidia-smi`. If batches spend most of their time waiting for data, do not launch a multi-epoch full run yet; first inspect mount/cache behavior and run a bounded throughput pilot.

VFS cache capacity is a ceiling, not a guarantee that the entire selected dataset fits. If the raw selected files exceed 256 GiB, older entries can be evicted and read again in later epochs. Monitor cache and filesystem space with:

```bash
df -h "$HOME/raid/Clara/relight_runtime/rclone_vfs"
du -sh "$HOME/raid/Clara/relight_runtime/rclone_vfs"
```

## Local project copy (30k subset)

To avoid repeatedly reading sample files over the mount, stage a deterministic 30,000-sample selection from the existing `xxdrive:` remote into the ignored project folder `data/synthhuman_local30000/raw/` (about 120,000 files: RGB, alpha, depth, normal). The exact selected manifest and file list are saved inside the project checkout. Check free disk space there before starting; raw source sizes vary.

```bash
./scripts/stage_dgx_subset.sh
```

If SSH disconnects or transfer stops, run the same command again. `rclone copy` reuses files already copied; the script verifies all expected files are non-empty and decodes one sample before reporting success. Then run audit/training using `configs/dgx_fastvit_t8_depth_normals_local30000_seed42.yaml`. This path uses rclone only to stage; training reads the local copy. Keep the repository and staged dataset in the project/work folder, not under `/raid`. The `/raid` VFS cache remains useful for the separate streaming workflow.
