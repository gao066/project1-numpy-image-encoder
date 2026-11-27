"""Extra robustness evidence: batch sizes, orders, dtypes, image modes."""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import ROOT as ASSET_ROOT, load_weights, load_images, image_records  # noqa: E402
from image_encoder import ImageEncoder  # noqa: E402

enc = ImageEncoder(load_weights())
images = load_images()
ids = [r['id'] for r in image_records()]

# 1) L2 norm of every output row
full = enc(images)
norms = np.sqrt((full * full).sum(-1))
print('norm min/max      :', float(norms.min()), float(norms.max()))
print('dtype/shape       :', full.dtype, full.shape)

# 2) single image equals its row in the full batch
one = enc(images[3:4])
print('B=1 max_abs diff  :', float(np.abs(one[0] - full[3]).max()))

# 3) permuted batch equals permuted rows
order = [7, 2, 9, 0]
perm = enc(images[order])
print('permuted max diff :', float(np.abs(perm - full[order]).max()))

# 4) odd-size / grayscale / RGBA inputs through the supplied preprocessing
tmp = Path(tempfile.mkdtemp(prefix='extra_verify_'))
src = Image.open(ASSET_ROOT / 'assets' / 'images' / 'image_01.jpg').convert('RGB')
src.resize((301, 457)).save(tmp / 'odd_size.png')
src.convert('L').save(tmp / 'gray.png')
src.convert('RGBA').save(tmp / 'alpha.png')
mixed = load_images([str(tmp / 'odd_size.png'), str(tmp / 'gray.png'), str(tmp / 'alpha.png')])
out = enc(mixed)
print('mixed modes shape :', mixed.shape, '->', out.shape,
      'finite:', bool(np.isfinite(out).all()))

# 5) patchify contract: manual extraction of one patch agrees
p = enc.patchify(images[:1])[0]
manual = images[0][:, 32:64, 96:128].reshape(-1)
print('patch(1,3) diff   :', float(np.abs(p[1 * 7 + 3] - manual).max()))
