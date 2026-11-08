"""attnscope: attention / residual-stream probes for the COS vs DICT vs CONTROL experiment.

``attnscope.plotting`` needs only numpy/matplotlib/seaborn.
``attnscope.forward`` and ``attnscope.scope`` need MLX (Apple Silicon).
They are not imported here so the plotting code works on any machine.
"""

# Where each part of the original summer code ended up:
#   TylerScope.manually_run_foward, tokenize_input   -> attnscope/forward.py
#   TylerScope.gen_from_prompt, static_cam_at_end_conv,
#     dynamic_camera_atten_slice                     -> attnscope/scope.py
#   final_lamma_vis.ipynb (final figures + GIF), the max-head and
#     specialization figures from llama_vis.ipynb    -> attnscope/plotting.py
#   (new, no original)                               -> attnscope/analysis.py
# Each file starts with a mapping table, and comments inside each function
# quote the original line(s) every new line replaces.

# Names exposed by `from attnscope import *`. Nothing is imported eagerly, so
# `import attnscope.plotting` works without MLX installed.
__all__ = ["forward", "scope", "plotting"]
