"""Run the whole HFRI v2 pipeline in order:  python3 run_all.py
Total time on a laptop: about 12-15 minutes (step 06 is the slow one)."""
import subprocess
import sys
import time
from pathlib import Path

STEPS = ["01_download_data", "02_simulation", "03_build_master", "04_validation", "05_prospective",
         "06_weights", "07_savings", "08_benchmarks", "09_bands", "10_figures_tables"]
code = Path(__file__).resolve().parent / "code"
start = time.time()
for step in STEPS:
    print(f"\n========== {step} ==========", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, "-u", str(code / f"{step}.py")], cwd=code)
    if r.returncode != 0:
        sys.exit(f"{step} failed - fix the error above, then run again.")
    print(f"---------- {step} done in {time.time() - t:.0f}s", flush=True)
print(f"\nAll done in {(time.time() - start) / 60:.1f} minutes. Results: output/")
