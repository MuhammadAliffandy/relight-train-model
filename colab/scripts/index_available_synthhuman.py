# Colab Drive API inventory helper. Active DGX training indexes the mounted raw folder through rclone.
# Requires the runtime settings, Drive paths, and Drive API setup cells first.
from tqdm.auto import tqdm
import re

if DATASET_MODE != 'pilot':
    raise ValueError('Keep DATASET_MODE="pilot" for this experiment.')
INDEX_PATH = DATA_ROOT / 'pilot_manifests' / 'synthhuman_inventory_v3.json'
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
index_config = {'version': 3, 'folder_id': SOURCE_FOLDER_ID}
state = {'config': index_config, 'files': {}, 'folders': [SOURCE_FOLDER_ID],
         'folder_position': 0, 'page_token': None, 'enumeration_done': False,
         'scanned': 0, 'ambiguous_names': [], 'empty_file_ids': []}
if INDEX_PATH.is_file():
    saved = json.loads(INDEX_PATH.read_text())
    if saved.get('config') == index_config:
        state = saved
all_files = state['files']
seen_folders = set(state['folders'])
ambiguous_names = set(state['ambiguous_names'])
empty_file_ids = set(state['empty_file_ids'])
pattern = re.compile(r'^(rgb|alpha|depth|normal)_(\d+)\.(png|exr)$')
required = {'rgb', 'alpha', 'depth', 'normal'}

def save_inventory():
    state['ambiguous_names'] = sorted(ambiguous_names)
    state['empty_file_ids'] = sorted(empty_file_ids)
    temporary = INDEX_PATH.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state))
    temporary.replace(INDEX_PATH)

if not state['enumeration_done']:
    pages = 0
    progress = tqdm(desc='Inventory existing Drive files + subfolders', unit='entry', initial=state['scanned'])
    try:
        while state['folder_position'] < len(state['folders']):
            folder_id = state['folders'][state['folder_position']]
            kwargs = {'q': f"'{drive_query_literal(folder_id)}' in parents and trashed = false",
                      'pageSize': 1000, 'spaces': 'drive',
                      'fields': 'nextPageToken,incompleteSearch,files(id,name,size,mimeType)'}
            if state['page_token']:
                kwargs['pageToken'] = state['page_token']
            try:
                response = DRIVE_API.files().list(**kwargs).execute(num_retries=5)
            except HttpError as error:
                if error.resp.status == 400 and state['page_token']:
                    state['page_token'] = None
                    save_inventory()
                    print('Expired page token: restarting this folder, keeping indexed file IDs.')
                    continue
                raise
            if response.get('incompleteSearch'):
                raise RuntimeError('Drive returned an incomplete search; retry indexing before selecting training data.')
            items = response.get('files', [])
            for item in items:
                if item.get('mimeType') == 'application/vnd.google-apps.folder':
                    if item['id'] not in seen_folders:
                        state['folders'].append(item['id'])
                        seen_folders.add(item['id'])
                    continue
                name = item['name']
                match = pattern.fullmatch(name)
                if not match:
                    continue
                kind, _, extension = match.groups()
                if extension != ('png' if kind in ('rgb', 'alpha') else 'exr'):
                    continue
                size = int(item.get('size', 0))
                if size <= 0:
                    empty_file_ids.add(item['id'])
                    continue
                existing = all_files.get(name)
                if existing and existing['id'] != item['id']:
                    ambiguous_names.add(name)
                    continue
                all_files[name] = {'id': item['id'], 'size': size}
            state['scanned'] += len(items)
            progress.update(len(items))
            state['page_token'] = response.get('nextPageToken')
            if not state['page_token']:
                state['folder_position'] += 1
            progress.set_postfix(files=len(all_files), folder=f"{state['folder_position']}/{len(state['folders'])}")
            pages += 1
            if pages % 100 == 0:
                save_inventory()
        state['enumeration_done'] = True
    finally:
        save_inventory()
        progress.close()

# Select replacements from complete, unambiguous sample groups actually found.
groups = {}
ambiguous_samples = set()
for name, metadata in all_files.items():
    kind, sample_id, _ = pattern.fullmatch(name).groups()
    sample_id = int(sample_id)
    groups.setdefault(sample_id, {})[kind] = name
    if name in ambiguous_names:
        ambiguous_samples.add(sample_id)
complete_ids = sorted(sample_id for sample_id, kinds in groups.items()
                      if required <= kinds.keys() and sample_id not in ambiguous_samples)
incomplete = [{'sample_id': sample_id, 'missing': sorted(required - kinds.keys())}
              for sample_id, kinds in groups.items() if not required <= kinds.keys()]
report = {'files_indexed': len(all_files), 'folders_found': len(state['folders']),
          'complete_samples': len(complete_ids), 'incomplete_samples': len(incomplete),
          'ambiguous_samples': len(ambiguous_samples), 'empty_files': len(empty_file_ids),
          'incomplete_examples': incomplete[:100], 'ambiguous_examples': sorted(ambiguous_samples)[:100]}
report_path = INDEX_PATH.with_name('synthhuman_inventory_report_v3.json')
report_path.write_text(json.dumps(report, indent=2))
print(f"Complete samples: {len(complete_ids):,}; skipped incomplete: {len(incomplete):,}; ambiguous IDs: {len(ambiguous_samples):,}; empty files: {len(empty_file_ids):,}")
print('Inventory report:', report_path)
if len(complete_ids) < PILOT_SAMPLE_COUNT:
    raise ValueError(f'Only {len(complete_ids):,} complete samples exist in this inventory, below the requested {PILOT_SAMPLE_COUNT:,}. Check SOURCE_FOLDER_ID or set PILOT_SAMPLE_COUNT to an available count; missing labels cannot be fabricated.')
selected_ids = sorted(random.Random(SEED).sample(complete_ids, PILOT_SAMPLE_COUNT))
PILOT_FILE_IDS = {}
manifest_rows = []
for number in selected_ids:
    kinds = groups[number]
    for name in kinds.values():
        PILOT_FILE_IDS[name] = all_files[name]
    manifest_rows.append({'image': str(SYNTHHUMAN_ROOT / kinds['rgb']),
                          'depth': str(SYNTHHUMAN_ROOT / kinds['depth']),
                          'normal': str(SYNTHHUMAN_ROOT / kinds['normal']),
                          'mask': str(SYNTHHUMAN_ROOT / kinds['alpha']),
                          'subject_id': f'sample_{number:07d}', 'label_source': 'SynthHuman/DAViD renderer',
                          'depth_units': 'm', 'depth_scale_to_m': 0.01, 'normal_convention': 'camera_xyz_signed'})
# Separate the new sample selection from any previous pilot checkpoints/manifests.
MANIFEST = DATA_ROOT / 'pilot_manifests' / f'synthhuman_available_{PILOT_SAMPLE_COUNT}_seed{SEED}_v3.jsonl'
OUTPUT_DIR = DRIVE_ROOT / 'runs' / f'fastvit_t8_fpn_depth_normals_available{PILOT_SAMPLE_COUNT}_seed{SEED}_driveapi_v3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
temporary = MANIFEST.with_suffix('.jsonl.tmp')
temporary.write_text(''.join(json.dumps(row) + '\n' for row in manifest_rows))
temporary.replace(MANIFEST)
print(f'Selected {len(manifest_rows):,} complete samples using seed {SEED}; missing groups were replaced from available data.')
print('Manifest:', MANIFEST)
print('New run directory:', OUTPUT_DIR)
# Keep only selected file metadata in RAM after the full inventory has been persisted.
del all_files, groups, state
if 'saved' in globals():
    del saved
