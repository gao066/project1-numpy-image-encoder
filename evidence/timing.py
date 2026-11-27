"""Wall-clock timing of the ten-image forward pass on CPU (no GPU, no cache)."""
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import load_images, load_weights  # noqa: E402
from image_encoder import ImageEncoder  # noqa: E402

t0 = time.perf_counter()
weights = load_weights()
images = load_images()
load_seconds = time.perf_counter() - t0

enc = ImageEncoder(weights)

t0 = time.perf_counter()
out = enc(images)
first = time.perf_counter() - t0

t0 = time.perf_counter()
enc(images)
second = time.perf_counter() - t0

print('python       :', platform.python_version())
print('numpy        :', np.__version__)
print('images       :', images.shape, images.dtype)
print('output       :', out.shape, out.dtype)
print('asset load   : %.2f s' % load_seconds)
print('forward #1   : %.2f s (%.2f s/image)' % (first, first / len(images)))
print('forward #2   : %.2f s (%.2f s/image)' % (second, second / len(images)))
print('forward mean : %.2f s' % ((first + second) / 2))
