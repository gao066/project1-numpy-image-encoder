"""Quantify two typical implementation errors against the block0 reference."""
import sys, shutil, importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0, r'C:/Users/11593/Desktop/project1-学生版/project1-numpy-student')
from runtime import ROOT, load_weights, load_images

SRC = Path(r'C:/Users/11593/Desktop/project1-学生版/project1-numpy-student/image_encoder.py')
text = SRC.read_text(encoding='utf-8')
weights = load_weights()
images = load_images()[:2]
with np.load(ROOT / 'assets/reference_outputs.npz', allow_pickle=False) as ref:
    target = ref['block0_tokens'][:2]


def run(module_name, source):
    path = Path('variant_' + module_name + '.py')
    path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    enc = module.ImageEncoder(weights)
    return enc.blocks[0](enc.pre_norm(enc.embed(images)))


# Error A: merge heads with a direct reshape (no transpose back).
bad_a = text.replace(
    "merged = context.transpose(0, 2, 1, 3).reshape(",
    "merged = context.reshape(")
out_a = run('bad_merge', bad_a)
print('A missing transpose : max_abs=%.3e mean_abs=%.3e shape=%s' %
      (np.abs(out_a - target).max(), np.abs(out_a - target).mean(), out_a.shape))

# Error B: divide by a float64 scale, which upcasts every later array.
bad_b = text.replace("scale = np.float32(np.sqrt(self.head_dim))",
                     "scale = np.sqrt(self.head_dim)")
out_b = run('bad_dtype', bad_b)
print('B float64 scale     : dtype=%s (check.py requires float32) max_abs=%.3e' %
      (out_b.dtype, np.abs(out_b - target).max()))
