# Paste into a new Colab cell after preprocessing and before shard preparation.
# Install the decoder first in another cell: %pip install "OpenEXR>=3.3,<4"
import os
from pathlib import Path
import numpy as np
from PIL import Image
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def read_exr_array(path, kind=None):
    path = Path(path)
    with path.open('rb') as handle:
        prefix = handle.read(16)
    if prefix[:4] != b'\x76\x2f\x31\x01':
        raise ValueError(f'{path.name} does not contain an OpenEXR header: first16_hex={prefix.hex()}, bytes={path.stat().st_size}. This is a source-file format/content problem, not a resize problem.')
    try:
        import OpenEXR
    except ImportError:
        import cv2
        try:
            array = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        except cv2.error as error:
            raise RuntimeError('OpenCV cannot decode this EXR. Install OpenEXR with: %pip install "OpenEXR>=3.3,<4"') from error
        if array is None:
            raise RuntimeError(f'OpenCV returned None for {path.name}. Install OpenEXR with: %pip install "OpenEXR>=3.3,<4"')
        array = array[..., ::-1] if array.ndim == 3 else array
        if kind == 'depth' and array.ndim == 3:
            array = array[..., :1]
        return array.astype(np.float32)
    if not hasattr(OpenEXR, 'File'):
        raise RuntimeError('This reader requires OpenEXR >=3.3. Update the package before retrying.')
    try:
        with OpenEXR.File(str(path), separate_channels=True) as image:
            if len(image.parts) != 1:
                raise ValueError(f'{path.name}: multipart EXR is not supported for this label contract.')
            channels = image.channels()
            lookup = {name.upper(): name for name in channels}
            if kind == 'normal':
                if {'R', 'G', 'B'} <= lookup.keys():
                    names = [lookup[x] for x in ('R', 'G', 'B')]
                elif {'X', 'Y', 'Z'} <= lookup.keys():
                    names = [lookup[x] for x in ('X', 'Y', 'Z')]
                else:
                    raise ValueError(f'{path.name}: normals require R/G/B or X/Y/Z channels; found {sorted(channels)}.')
            elif kind == 'depth':
                if 'Z' in lookup:
                    names = [lookup['Z']]
                elif len(channels) == 1:
                    names = [next(iter(channels))]
                elif 'R' in lookup:
                    names = [lookup['R']]  # Preserves the prior RGB-depth red-channel convention.
                else:
                    raise ValueError(f'{path.name}: ambiguous depth channels: {sorted(channels)}.')
            else:
                raise ValueError('Declare EXR kind as depth or normal.')
            planes = []
            for name in names:
                pixels = channels[name].pixels
                if pixels.ndim != 2 or pixels.dtype.kind not in 'fiu':
                    raise ValueError(f'{path.name}: channel {name} is not a flat numeric image.')
                planes.append(np.array(pixels, dtype=np.float32, copy=True))
            return np.stack(planes, axis=-1)
    except Exception as error:
        raise ValueError(f'OpenEXR decode failed for {path.name}: {error}') from error


def read_array(path, rgb=False, mask=False, exr_kind=None):
    # Preparation passes local temporary paths, so no Drive FUSE access occurs here.
    path = Path(path)
    if path.suffix.lower() == '.exr':
        arr = read_exr_array(path, kind=exr_kind)
    elif path.suffix.lower() == '.npy':
        arr = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            arr = np.asarray(image.convert('L' if mask else 'RGB'))
    if rgb:
        if arr.dtype != np.uint8:
            arr = np.clip(arr*255 if arr.max() <= 1 else arr, 0, 255).astype(np.uint8)
        arr = np.asarray(Image.fromarray(arr).convert('RGB').resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BILINEAR), dtype=np.float32)/255.0
    elif mask:
        arr = resize_array((arr.squeeze() > 0).astype(np.uint8)*255, INPUT_SIZE, True)[..., 0] > 0
    else:
        arr = resize_array(arr.astype(np.float32), INPUT_SIZE).astype(np.float32)
    return arr

print('EXR decoder patch active: named channels, header check, OpenEXR reader with OpenCV fallback.')
