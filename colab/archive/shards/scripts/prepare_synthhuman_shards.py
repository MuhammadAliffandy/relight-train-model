# Embedded in prepare_synthhuman_shards.ipynb after its local-path preprocessing cell.
# Use that notebook; the older direct-API training reader is not the preparation decoder.
import concurrent.futures
import copy
import hashlib
import io
import shutil
import tarfile
import tempfile
import threading
from tqdm.auto import tqdm

SHARD_SAMPLES = 250
FETCH_WORKERS = 8
SHARD_SCRATCH_ROOT = Path('/content')
PREPARE_SHARD_LIMIT = 2  # 2 shards = 500 preparation samples; use None for all 50k.
RUN_PREPARATION = False  # Set True when ready for reads from your existing Drive.
SHARD_ROOT = DATA_ROOT / 'prepared_shards' / f'synthhuman_{PILOT_SAMPLE_COUNT}_seed{SEED}_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_v1'
SHARD_MANIFEST_PATH = SHARD_ROOT / 'manifest.json'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_existing_drive_shards():
    active_reader = globals().get('read_array')
    if active_reader is not None and 'read_exr_array' not in active_reader.__code__.co_names:
        raise RuntimeError('The old OpenCV-only read_array is still active. Run activate_exr_decoder.py AFTER preprocessing, then rerun ONLY preparation.')
    if SHARD_SAMPLES < 1 or FETCH_WORKERS < 1:
        raise ValueError('SHARD_SAMPLES and FETCH_WORKERS must be positive.')
    if PREPARE_SHARD_LIMIT is not None and PREPARE_SHARD_LIMIT < 1:
        raise ValueError('PREPARE_SHARD_LIMIT must be positive or None.')
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    selection = [(part, row['subject_id']) for part in ('train', 'val', 'test') for row in parts[part]]
    config = {'format_version': 1, 'samples': len(selection), 'seed': SEED,
              'input_size': list(INPUT_SIZE), 'source_folder_id': SOURCE_FOLDER_ID,
              'selection_sha256': hashlib.sha256(json.dumps(selection).encode()).hexdigest(),
              'shard_samples': SHARD_SAMPLES, 'depth_units': 'm',
              'depth_dtype': 'float32', 'normal_dtype': 'float32',
              'rgb_dtype': 'uint8', 'mask_dtype': 'uint8',
              'preprocessing': 'geometry_dataset_v1_no_augmentation'}
    packed = {'config': config, 'shards': [], 'complete': False}
    if SHARD_MANIFEST_PATH.is_file():
        packed = json.loads(SHARD_MANIFEST_PATH.read_text())
        if packed['config'] != config:
            raise ValueError('Prepared dataset configuration differs. Use a new SHARD_ROOT instead of mixing datasets.')
    existing = {item['name']: item for item in packed['shards']}
    plan = []
    for part in ('train', 'val', 'test'):
        for offset in range(0, len(parts[part]), SHARD_SAMPLES):
            name = f'{part}_{offset // SHARD_SAMPLES:05d}.tar'
            plan.append((part, name, parts[part][offset:offset+SHARD_SAMPLES]))
    chosen = plan if PREPARE_SHARD_LIMIT is None else plan[:PREPARE_SHARD_LIMIT]
    todo = []
    for part, name, batch_rows in chosen:
        item = existing.get(name)
        path = SHARD_ROOT / name
        if item and path.is_file() and path.stat().st_size == item['bytes']:
            if [r['subject_id'] for r in item['rows']] != [r['subject_id'] for r in batch_rows]:
                raise ValueError(f'Sample membership changed for {name}.')
        else:
            todo.append((part, name, batch_rows))

    def save_manifest():
        packed['shards'] = [existing[name] for _, name, _ in plan if name in existing]
        packed['complete'] = len(packed['shards']) == len(plan)
        temporary = SHARD_MANIFEST_PATH.with_suffix('.json.part')
        temporary.write_text(json.dumps(packed))
        temporary.replace(SHARD_MANIFEST_PATH)

    save_manifest()
    print(f'Planned dataset: {len(selection):,} samples / {len(plan)} shards. This run: {sum(len(x[2]) for x in todo):,} new samples.')
    client_local = threading.local()

    def fetch(path, destination):
        from googleapiclient.discovery import build
        import google_auth_httplib2, httplib2
        if not hasattr(client_local, 'client'):
            client_local.client = build('drive', 'v3', http=google_auth_httplib2.AuthorizedHttp(
                copy.copy(credentials), http=httplib2.Http(timeout=120)), cache_discovery=False)
        name = Path(path).name
        metadata = PILOT_FILE_IDS[name]
        content = client_local.client.files().get_media(fileId=metadata['id']).execute(num_retries=5)
        if len(content) != metadata['size']:
            raise IOError(f'Incomplete API read: {name}')
        destination.write_bytes(content)
        return len(content)

    started = time.perf_counter()
    done_samples = 0
    uploaded_bytes = 0
    progress = tqdm(total=sum(len(x[2]) for x in todo), desc='Prepare existing dataset', unit='sample')
    try:
        with tempfile.TemporaryDirectory(dir=SHARD_SCRATCH_ROOT, prefix='synthhuman_pack_') as scratch, concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            scratch = Path(scratch)
            for part, name, shard_rows in todo:
                shard_start = time.perf_counter()
                archive_path = scratch / name
                prepared_rows = []
                with tarfile.open(archive_path, 'w') as archive:
                    for start in range(0, len(shard_rows), 8):
                        source_rows = shard_rows[start:start+8]
                        expected_bytes = sum(PILOT_FILE_IDS[Path(r[k]).name]['size'] for r in source_rows for k in ('image', 'depth', 'normal', 'mask'))
                        if shutil.disk_usage(scratch).free < expected_bytes + 1024**3:
                            raise OSError('Insufficient local scratch space. Completed shards are retained on Drive.')
                        with tempfile.TemporaryDirectory(dir=scratch, prefix='raw_') as raw:
                            raw = Path(raw)
                            local_rows = []
                            futures = []
                            for row in source_rows:
                                local = dict(row)
                                for key in ('image', 'depth', 'normal', 'mask'):
                                    target = raw / Path(row[key]).name
                                    futures.append(pool.submit(fetch, row[key], target))
                                    local[key] = str(target)
                                local_rows.append(local)
                            try:
                                for future in concurrent.futures.as_completed(futures):
                                    future.result()
                            finally:
                                for future in futures:
                                    future.cancel()
                                concurrent.futures.wait(futures)
                            dataset = RelightingDataset(local_rows, augment=False)
                            for index, row in enumerate(source_rows):
                                image, depth, normal, mask = dataset[index]
                                if not all(torch.isfinite(t).all() for t in (image, depth, normal, mask)) or mask.sum() <= 0:
                                    raise ValueError(f'Invalid decoded sample: {row["subject_id"]}. No placeholder label was written.')
                                # Store resized RGB without ImageNet normalization; geometric maps keep float32 precision.
                                rgb = ((image.numpy() * np.array([.229,.224,.225], np.float32)[:,None,None] + np.array([.485,.456,.406], np.float32)[:,None,None]) * 255).round().clip(0,255).astype(np.uint8)
                                member = row['subject_id'] + '.npz'
                                payload = io.BytesIO()
                                np.savez_compressed(payload, rgb=rgb, depth=depth.numpy().astype(np.float32), normal=normal.numpy().astype(np.float32), mask=mask.numpy().astype(np.uint8))
                                data = payload.getvalue()
                                info = tarfile.TarInfo(member)
                                info.size = len(data)
                                info.mtime = 0
                                archive.addfile(info, io.BytesIO(data))
                                prepared_rows.append({**row, 'partition': part, 'shard': name, 'member': member})
                                progress.update(1)
                                done_samples += 1
                                progress.set_postfix(shard=name, samples_s=f'{done_samples/max(time.perf_counter()-started,1e-9):.2f}')
                checksum = file_sha256(archive_path)
                size = archive_path.stat().st_size
                remote_temporary = SHARD_ROOT / (name + '.part')
                shutil.copyfile(archive_path, remote_temporary)
                if remote_temporary.stat().st_size != size:
                    raise IOError(f'Incomplete shard upload: {name}')
                remote_temporary.replace(SHARD_ROOT / name)
                existing[name] = {'name': name, 'partition': part, 'bytes': size, 'sha256': checksum, 'rows': prepared_rows}
                save_manifest()
                archive_path.unlink()
                uploaded_bytes += size
                print(f'{name}: {len(prepared_rows)} samples saved to Drive; {size/1024**2:.1f} MiB; {time.perf_counter()-shard_start:.1f}s')
    finally:
        progress.close()
    elapsed = time.perf_counter() - started
    if done_samples:
        print(f'Preparation measurement: {done_samples} samples in {elapsed/60:.1f} minutes; {uploaded_bytes/1024**2/done_samples:.2f} MiB/sample packed.')
        print(f'Extrapolation for {len(selection):,} samples: {elapsed/done_samples*len(selection)/3600:.1f} hours preparation; distribution/network can change this.')
    print('Ready for full training:', packed['complete'])
    print('Manifest:', SHARD_MANIFEST_PATH)
    if not packed['complete']:
        print('To continue all 50k samples, set PREPARE_SHARD_LIMIT=None and rerun this cell. Finished shards are skipped.')

if RUN_PREPARATION:
    prepare_existing_drive_shards()
else:
    print('Preparation disabled. Set RUN_PREPARATION=True for the first 500 samples; PREPARE_SHARD_LIMIT=None continues the full selection.')
