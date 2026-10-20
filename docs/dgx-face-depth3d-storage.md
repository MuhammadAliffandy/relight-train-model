# Face-Depth-3D storage on DGX

Observed on 2026-10-20:

- The successful earlier V9.1 checkpoint transfer used `cp` into `$HOME/xxdrive`, a mounted Google Drive path.
- The `relight-dgx` shell reported `rclone: command not found`. The mount process shows that a binary exists at `$HOME/miniconda3/envs/brain_env/bin/rclone`; it is outside the `relight-dgx` PATH. Use its absolute path instead of installing another copy or changing the Python environment.
- `mountpoint -q "$HOME/xxdrive"` succeeded, but reading `AppleCapstoneRelighting/datasets/face_depth_3d` returned `OSError: [Errno 5] Input/output error`. A mountpoint check alone does not establish that the remote is healthy.
- `findmnt` identified `$HOME/xxdrive` as `fuse.rclone`, with an `rclone mount xxdrive:` process running. A separate `clara_drive:` mount also exists; do not confuse its remote with `xxdrive:`.
- Direct access through the `brain_env` rclone binary subsequently failed with `invalid_grant: maybe token expired?`. This is an OAuth authorization failure for `xxdrive:`, not a missing Parquet file. Reconnect that remote before retrying either copy path.

Before starting the Face-Depth-3D preparation job, run:

```bash
findmnt -M "$HOME/xxdrive" -o TARGET,SOURCE,FSTYPE
ls -ld "$HOME/xxdrive/AppleCapstoneRelighting/datasets/face_depth_3d"
find "$HOME/xxdrive/AppleCapstoneRelighting/datasets/face_depth_3d" \
  -maxdepth 1 -type f -name '*.parquet' | wc -l
```

The last command should report 22 files. If `ls` returns an I/O error, do not restart the copy job yet. Inspect the mount type and process that created it, then restore that mount using its original method. Avoid unmounting a shared drive while another job is using it.

When the FUSE mount fails but the remote itself is reachable, bypass the mount:

```bash
RCLONE="$HOME/miniconda3/envs/brain_env/bin/rclone"
REMOTE="xxdrive:AppleCapstoneRelighting/datasets/face_depth_3d"
"$RCLONE" lsf "$REMOTE" --files-only
nohup python -u scripts/prepare_face_depth3d_dgx.py \
  --rclone-remote "$REMOTE" --rclone-binary "$RCLONE" \
  > logs/face_depth3d_prepare.log 2>&1 < /dev/null &
echo $! > logs/face_depth3d_prepare.pid
```

Only start the `nohup` job if `lsf` lists the 22 expected Parquet files. The direct copy leaves the existing mounts untouched.

If `lsf` reports `invalid_grant`, re-authorize the existing remote interactively:

```bash
RCLONE="$HOME/miniconda3/envs/brain_env/bin/rclone"
"$RCLONE" config file
"$RCLONE" config reconnect xxdrive:
```

On a headless DGX session, answer `n` when asked to use a browser on the server. Follow the `rclone authorize` command printed by the reconnect flow on a computer with a browser, then paste the resulting token into the DGX prompt. Keep the token and rclone config out of logs and chat. After reconnect, retry `"$RCLONE" lsf "$REMOTE" --files-only` and start the direct-copy preparation job only when it lists all 22 files. If the old FUSE process still returns I/O errors, continue using the direct-copy route rather than relying on that process.

Once the mount is readable, use the original preparation path:

```bash
nohup python -u scripts/prepare_face_depth3d_dgx.py \
  > logs/face_depth3d_prepare.log 2>&1 < /dev/null &
echo $! > logs/face_depth3d_prepare.pid
```

The job can be rerun after interruption. Complete Parquet files are reused, partial copies resume, and extraction files are written atomically. The optional `--rclone-remote` path requires a working `rclone` executable and configuration; it is not the default DGX path.
