"""Extra robustness evidence: batch sizes, orders, dtypes, image modes."""
from pathlib import Path
import numpy as np
from PIL import Image
from runtime import ROOT, load_weights, load_images, image_records
from image_encoder import ImageEncoder

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
tmp = Path('C:/Users/11593/WorkBuddy/2026-11-26-13-11-37/report_assets/tmp_inputs')
tmp.mkdir(parents=True, exist_ok=True)
src = Image.open(ROOT / 'assets' / 'images' / 'image_01.jpg').convert('RGB')
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
