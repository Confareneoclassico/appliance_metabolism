"""Rerun points of the sample with their seeds redrawn, for the total index of
the seeds on the model itself.

    python3 -m gsa.reseed --bundle BUNDLE --base RUNS --out RESEED --which seeds
                          [--points 256 | --list 0,1,2409] [--workers W] [--threads T]

Variants (--which):

  seeds    the seed of the country multipliers (X8) and the seed of K-means
           (X21) both redrawn
  kmeans   the seed of K-means (X21) alone redrawn
  same     nothing redrawn: the points run again as they are, a check that the
           model gives the same outputs as in RUNS (it is deterministic)

Run i of RESEED/<which> is point i of the sample of RUNS with the seeds of the
variant redrawn from a stream of its own, np.random.default_rng([RESEED_SEED,
variant, i]), so that a point gets the same new seeds whichever other points
are run. The estimator of Jansen (1999), T = mean((y_i - y'_i)^2) / (2 Var Y),
then gives the total index of the redrawn seeds on every output (gsa.analyse,
part seeds): for the seeds together it is the share of the variance that no
function of the other factors can reproduce, which closes the variance budget
of the surrogate's Shapley effects.

The settings come from RUNS/design.json: the bundle (its creation time and
size), n_init, the law of the powers and the library versions must match those
of the base runs, or the run is refused. Files are written as by gsa.run, so
an interrupted variant resumes, and gsa.collect gathers a variant folder into
files that can be shared like those of the main sample.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsa import design, run as gsa_run  # noqa: E402

RESEED_SEED = 20261124
VARIANTS = {"seeds": ["X8", "X21"], "kmeans": ["X21"], "same": []}
CODE = {"seeds": 1, "kmeans": 2, "same": 0}


def redrawn(U, points, which):
    """Coordinates of the given points of U with the seeds of the variant
    redrawn (the other columns unchanged)."""
    cols = [design.COLUMNS.index(c) for c in VARIANTS[which]]
    out = np.array(U[points], dtype=float)
    for k, i in enumerate(points):
        if cols:
            out[k, cols] = np.random.default_rng([RESEED_SEED, CODE[which], int(i)]).random(len(cols))
    return out


def base_settings(base, bundle):
    """Settings of the base runs, checked against the bundle and this
    environment."""
    dpath = Path(base) / "design.json"
    if not dpath.is_file():
        sys.exit(f"no design.json in {base}: give the folder of the runs of the main sample")
    have = json.loads(dpath.read_text())
    meta = json.loads((Path(bundle) / "meta.json").read_text())
    if (have.get("bundle_created"), have.get("household_days")) != (meta.get("created"),
                                                                   meta.get("household_days")):
        sys.exit(f"{bundle} is not the bundle of the runs in {base} (built {meta.get('created')} "
                 f"with {meta.get('household_days')} household-days, against "
                 f"{have.get('bundle_created')} with {have.get('household_days')})")
    versions = gsa_run.library_versions()
    if "versions" in have and have["versions"] != versions:
        sys.exit(f"the runs in {base} were made with "
                 + ", ".join(f"{k} {v}" for k, v in have["versions"].items())
                 + ", and this environment has "
                 + ", ".join(f"{k} {v}" for k, v in versions.items())
                 + ": activate the environment of the earlier runs")
    return have, versions


def compare_same(base, folder, points):
    """Largest absolute difference of every output between the reruns and
    the base runs."""
    worst = {}
    for i in points:
        a = json.loads((Path(base) / f"run_{i:05d}.json").read_text())["outputs"]
        b = json.loads((Path(folder) / f"run_{i:05d}.json").read_text())["outputs"]
        for k, v in a.items():
            if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
                worst[k] = max(worst.get(k, 0.0), abs(float(v) - float(b[k])))
            elif v != b.get(k):
                worst[k] = float("inf")
    return worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="folder of the runs of the main sample")
    parser.add_argument("--out", type=Path, required=True, help="folder of the reruns, one subfolder per variant")
    parser.add_argument("--which", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--points", type=int, default=256, help="rerun the first N points (default 256)")
    parser.add_argument("--list", default=None, help="rerun these points instead, e.g. 0,1,2409")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1, help="threads per worker")
    args = parser.parse_args(argv)

    have, versions = base_settings(args.base, args.bundle)
    m = int(have["m"])
    points = ([int(p) for p in args.list.split(",")] if args.list else list(range(args.points)))
    if max(points) >= 2 ** m:
        sys.exit(f"the sample of {args.base} has {2 ** m} points")
    missing = [i for i in points if not (args.base / f"run_{i:05d}.json").is_file()]
    if missing:
        sys.exit(f"{len(missing)} of the points have no run in {args.base} (first: {missing[0]})")
    folder = args.out / args.which
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {**{k: v for k, v in have.items() if k != "versions"}, "versions": versions,
              "bundle": str(Path(args.bundle).resolve()), "reseed": args.which,
              "redrawn": VARIANTS[args.which], "reseed_seed": RESEED_SEED,
              "base": str(Path(args.base).resolve())}
    dpath = folder / "design.json"
    if dpath.is_file():
        old = json.loads(dpath.read_text())
        differ = [k for k in wanted if k not in ("bundle", "base") and old.get(k) != wanted[k]]
        if differ:
            sys.exit(f"{folder} holds reruns made with other settings ({', '.join(differ)}): "
                     "use another folder")
    dpath.write_text(json.dumps(wanted, indent=1))

    U = design.sample(m, seed=have.get("sample_seed", gsa_run.SAMPLE_SEED))
    Unew = redrawn(U, points, args.which)
    xs = design.transform(Unew, uniform_in_watts=bool(have.get("uniform_in_watts")))
    todo = [k for k, i in enumerate(points) if not all(p.is_file() for p in gsa_run.run_paths(folder, i))]
    print(f"Variant {args.which} ({', '.join(VARIANTS[args.which]) or 'nothing'} redrawn): "
          f"{len(points) - len(todo)} of {len(points)} points already done; {len(todo)} to run with "
          f"{args.workers} worker(s) of {args.threads} thread(s), n_init = {have['n_init']}")
    t0, times, failed = time.time(), [], []

    def progress(i, seconds, out):
        times.append(seconds)
        left = (time.time() - t0) / len(times) * (len(todo) - len(times))
        if out is None:
            failed.append(i)
            print(f"point {i}: FAILED, see {folder}/run_{i:05d}.error", flush=True)
            return
        print(f"point {i}: {seconds:.0f} s; {len(times)} of {len(todo)} done, about "
              f"{gsa_run.hms(left)} left", flush=True)

    if todo:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        if args.workers == 1:
            from threadpoolctl import threadpool_limits
            gsa_run.init_worker(args.bundle, have["n_init"])
            with threadpool_limits(args.threads):
                for k in todo:
                    progress(*gsa_run.one_run(points[k], xs[k], Unew[k], folder))
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                     initializer=gsa_run.init_worker,
                                     initargs=(args.bundle, have["n_init"], args.threads)) as pool:
                futures = [pool.submit(gsa_run.one_run, points[k], xs[k], Unew[k], folder) for k in todo]
                for f in as_completed(futures):
                    progress(*f.result())
        print(f"{len(times)} reruns in {gsa_run.hms(time.time() - t0)}")
    if failed:
        print(f"{len(failed)} rerun(s) failed; run the same command again to retry them")
        return
    if args.which == "same":
        worst = compare_same(args.base, folder, points)
        off = {k: v for k, v in worst.items() if v > 1e-9}
        print("Reruns identical to the base runs" if not off else
              "Reruns differ from the base runs: " + ", ".join(f"{k} {v:.3g}" for k, v in off.items()))


if __name__ == "__main__":
    main()
