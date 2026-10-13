# Reader for prepared shards. Embedded in the training notebook preprocessing cell.
from collections import OrderedDict
import hashlib
import io
import shutil
import tarfile

SHARD_CACHE_GIB = 4
LOCAL_SHARD_CACHE = Path('/content/synthhuman_shard_cache') / PACKED_MANIFEST['config']['selection_sha256'][:16]
LOCAL_SHARD_CACHE.mkdir(parents=True, exist_ok=True)
SHARD_CACHE_BYTES = min(SHARD_CACHE_GIB * 1024**3, max(0, shutil.disk_usage('/content').free - 2*1024**3))
SHARD_RECORDS = {entry['name']: entry for entry in PACKED_MANIFEST['shards']}
SHARD_STATS = {'copies': 0, 'local_hits': 0, 'copied_bytes': 0, 'copy_seconds': 0.0}
for previous_handle in globals().get('shard_handles', {}).values():
    previous_handle.close()
shard_handles = OrderedDict()
shard_cache_bytes = 0
shard_cache_files = OrderedDict()
for existing_path in sorted(LOCAL_SHARD_CACHE.glob('*.tar'), key=lambda x: x.stat().st_mtime):
    record = SHARD_RECORDS.get(existing_path.name)
    if record and existing_path.stat().st_size == record['bytes']:
        shard_cache_files[existing_path.name] = record['bytes']
        shard_cache_bytes += record['bytes']
    else:
        existing_path.unlink()
if max(entry['bytes'] for entry in SHARD_RECORDS.values()) > SHARD_CACHE_BYTES:
    raise OSError('A prepared shard exceeds the available local cache budget.')


def shard_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def trim_shard_cache(required_bytes=0):
    global shard_cache_bytes
    while shard_cache_files and shard_cache_bytes + required_bytes > SHARD_CACHE_BYTES:
        oldest, size = shard_cache_files.popitem(last=False)
        handle = shard_handles.pop(oldest, None)
        if handle is not None:
            handle.close()
        (LOCAL_SHARD_CACHE / oldest).unlink(missing_ok=True)
        shard_cache_bytes -= size
trim_shard_cache()


def open_prepared_shard(name):
    global shard_cache_bytes
    if name in shard_handles:
        shard_cache_files.move_to_end(name)
        SHARD_STATS['local_hits'] += 1
        return shard_handles[name]
    record = SHARD_RECORDS[name]
    target = LOCAL_SHARD_CACHE / name
    started = time.perf_counter()
    cached = target.is_file() and target.stat().st_size == record['bytes'] and shard_sha256(target) == record['sha256']
    if not cached:
        if name in shard_cache_files:
            shard_cache_bytes -= shard_cache_files.pop(name)
        target.unlink(missing_ok=True)
        trim_shard_cache(record['bytes'])
        if shutil.disk_usage('/content').free < record['bytes'] + 2*1024**3:
            raise OSError('Insufficient local disk for the next shard. Clear unused local data and rerun.')
        temporary = target.with_suffix('.tar.part')
        try:
            shutil.copyfile(SHARD_ROOT / name, temporary)
            if temporary.stat().st_size != record['bytes'] or shard_sha256(temporary) != record['sha256']:
                raise IOError(f'Shard integrity check failed: {name}')
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        SHARD_STATS['copies'] += 1
        SHARD_STATS['copied_bytes'] += record['bytes']
        SHARD_STATS['copy_seconds'] += time.perf_counter() - started
        print(f'Staged {name}: {record["bytes"]/1024**2:.1f} MiB in {time.perf_counter()-started:.1f}s')
    if name not in shard_cache_files:
        shard_cache_files[name] = record['bytes']
        shard_cache_bytes += record['bytes']
    shard_cache_files.move_to_end(name)
    handle = tarfile.open(target, 'r:')
    handle.getmembers()
    shard_handles[name] = handle
    return handle


class RelightingDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        archive = open_prepared_shard(row['shard'])
        with archive.extractfile(row['member']) as member:
            with np.load(io.BytesIO(member.read()), allow_pickle=False) as sample:
                image = sample['rgb'].astype(np.float32) / 255.0
                depth = sample['depth'].astype(np.float32)
                normal = sample['normal'].astype(np.float32)
                mask = sample['mask'].astype(np.float32)
        expected = {'image': (3,*INPUT_SIZE), 'depth': (1,*INPUT_SIZE), 'normal': (3,*INPUT_SIZE), 'mask': (1,*INPUT_SIZE)}
        for key, array in [('image',image),('depth',depth),('normal',normal),('mask',mask)]:
            if array.shape != expected[key] or not np.isfinite(array).all():
                raise ValueError(f'Invalid prepared {key} for {row["subject_id"]}')
        if mask.sum() <= 0:
            raise ValueError(f'Empty prepared valid mask: {row["subject_id"]}')
        if self.augment and random.random() < .5:
            image,depth,normal,mask = [np.flip(a,axis=2).copy() for a in (image,depth,normal,mask)]
            normal[0] *= -1
        image = (image - np.array([.485,.456,.406],np.float32)[:,None,None]) / np.array([.229,.224,.225],np.float32)[:,None,None]
        tensors = [torch.from_numpy(a.copy()).float() for a in (image,depth,normal,mask)]
        tensors[2] = F.normalize(tensors[2],dim=0,eps=1e-6)
        return tuple(tensors)
print(f'Prepared-shard reader ready. Local cache cap: {SHARD_CACHE_BYTES/1024**3:.2f} GiB.')


def shard_sample_order(dataset, seed):
    grouped = {}
    for index, row in enumerate(dataset.rows):
        grouped.setdefault(row['shard'], []).append(index)
    rng = random.Random(seed)
    names = sorted(grouped)
    rng.shuffle(names)
    order = []
    for name in names:
        rng.shuffle(grouped[name])
        order.extend(grouped[name])
    return order
