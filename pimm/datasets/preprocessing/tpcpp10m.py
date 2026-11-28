"""Write TPCpp-10M's unlabeled point arrays uncompressed so that ``TPCpp10MDataset`` can memory-map them.

Run once after downloading the release:

    python -m pimm.datasets.preprocessing.tpcpp10m <data_root> [--workers N]

Each ``unlabeled/spacepoints_NNN.npz`` gets a ``spacepoints_NNN.data.npy`` next to it (1.4 GB each).
Shards already written are skipped, so an interrupted run can be restarted.
"""

import argparse
import glob
import os
import shutil
import zipfile
from multiprocessing import Pool


def unpack(path):
    out = path.replace(".npz", ".data.npy")
    if not os.path.exists(out):
        # The member's CRC is verified as it inflates, and the rename leaves only complete files under `out`.
        with zipfile.ZipFile(path) as archive, archive.open("data.npy") as src, open(out + ".tmp", "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        os.replace(out + ".tmp", out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_root", help="the release's root folder, holding labeled/ and unlabeled/")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    shards = sorted(glob.glob(os.path.join(args.data_root, "unlabeled", "spacepoints_*.npz")))
    if not shards:
        raise SystemExit(f"no unlabeled/spacepoints_*.npz under {args.data_root}")
    with Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(unpack, shards), 1):
            print(f"[{i}/{len(shards)}] {out}", flush=True)


if __name__ == "__main__":
    main()
