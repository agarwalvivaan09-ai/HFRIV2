"""01 - Download the three Federal Reserve data files, unzip them and check their fingerprints.

Files are saved in data/raw/. If a file is already there, it is not downloaded again.
A mismatching fingerprint means the Federal Reserve has re-issued the file; results may then differ slightly.
"""
import sys
import urllib.request
import zipfile

import pandas as pd

from hfri_common import RAW, SOURCES, sha256

ok = True
for key, src in SOURCES.items():
    target = RAW / src["file"]
    if not target.exists():
        zpath = RAW / src["zip"]
        if not zpath.exists():
            print(f"Downloading {src['name']} ...")
            req = urllib.request.Request(src["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(zpath, "wb") as f:
                f.write(r.read())
        with zipfile.ZipFile(zpath) as z:
            z.extract(src["file"], RAW)
    digest = sha256(target)
    match = digest == src["sha256"]
    ok &= match
    print(f"{src['file']:22s} SHA-256 {'OK' if match else 'DIFFERENT: ' + digest}")

# The panel is an Excel file; convert it to CSV once so later steps load quickly.
panel_csv = RAW / "SCFP2009panel.csv"
if not panel_csv.exists():
    print("Converting the panel Excel file to CSV (about a minute) ...")
    pd.read_excel(RAW / SOURCES["panel"]["file"]).to_csv(panel_csv, index=False)

print("All files match the paper." if ok else "WARNING: at least one file differs from the version used in the paper.")
sys.exit(0)
