# Extra pilot run: 1,000 all-clean runs to tighten the false-alarm interval.
import sys, io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import pilot
from pilot import run, wilson
S = 1000
any_flag = sum(run(0, s, poisoned=False)[1].any() for s in range(S))
lo, hi = wilson(any_flag, S)
print(f"all-clean runs with any contributor flagged: {any_flag}/{S} ({any_flag/S:.1%}); 95% Wilson CI {lo:.1%} to {hi:.1%}")
