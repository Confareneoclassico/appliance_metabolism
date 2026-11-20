"""Run the model over the sample of the sensitivity analysis.

    python3 -m gsa.run --bundle BUNDLE --runs RUNS [--m 10] [--first N]
                       [--workers W] [--threads T] [--n-init 10]
                       [--uniform-in-watts]

Run i is point i of the scrambled Sobol' sequence of gsa.design, so the first
2^k runs form the nested designs of the convergence analysis (128, 256, 512,
1,024). Each run writes RUNS/run_iiiii.json (factor values, coordinates,
outputs and seconds) and RUNS/run_iiiii.npz (the mean profiles by country and
day type, and their numbers of household-days). A run whose files exist is
skipped, so an interrupted sample resumes where it stopped, and --first N runs
only the first N points (a pilot). RUNS/design.json records the settings that
the results depend on; resuming with other settings is refused.

W workers each memory-map the bundle and use T threads. K-means takes most of
the time, so try a pilot of a few runs with, say, --workers 1 --threads 8 and
--workers 4 --threads 2, and keep the faster. Each worker needs about 1.5 GB
of memory for HETUS 2010.

The run files hold aggregates over groups of at least 50 household-days, but
they stay with the bundle; gsa.collect gathers what can be shared.
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

from gsa import design  # noqa: E402

SAMPLE_SEED = 20261119
MODEL = None


def settings(args, meta):
    return {"m": args.m, "sample_seed": SAMPLE_SEED, "uniform_in_watts": bool(args.uniform_in_watts),
            "n_init": args.n_init, "bundle": str(Path(args.bundle).resolve()),
            "bundle_created": meta.get("created"), "household_days": meta.get("household_days"),
            "factors": [f.id for f in design.FACTORS], "dummies": design.DUMMIES}


def jsonable(x):
    """Factor values with the power dicts keyed by text."""
    return {k: ({str(c): float(w) for c, w in v.items()} if isinstance(v, dict) else v)
            for k, v in x.items()}


def run_paths(runs, i):
    return Path(runs) / f"run_{i:05d}.json", Path(runs) / f"run_{i:05d}.npz"


def init_worker(bundle, n_init, threads=None):
    """Load the model in a worker, with at most `threads` threads for the
    rest of its life (None: leave the limits as they are)."""
    global MODEL
    if threads:
        from threadpoolctl import threadpool_limits
        globals()["_limits"] = threadpool_limits(threads)
    from gsa.model import Model
    MODEL = Model(bundle, n_init=n_init)


def one_run(i, x, u, runs):
    """One run, written to its files. A run that fails writes its traceback
    to run_iiiii.error instead (with no data in it) and is retried on resume."""
    t0 = time.time()
    js, npz = run_paths(runs, i)
    err = js.with_suffix(".error")
    try:
        out, prof, counts = MODEL.run(x)
    except Exception:
        import traceback
        err.write_text(traceback.format_exc())
        return i, time.time() - t0, None
    seconds = time.time() - t0
    if err.is_file():
        err.unlink()
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez(tmp, profiles=prof.astype(np.float32), counts=counts.astype(np.int32),
             countries=np.array([*(MODEL.meta["countries"]), "ALL"]))
    os.replace(tmp, npz)
    record = {"run": i, "seconds": seconds, "coordinates": [float(v) for v in u],
              "factors": jsonable(x), "outputs": out}
    tmp = js.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=float))
    os.replace(tmp, js)
    return i, seconds, out


def hms(s):
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="folder of the run files")
    parser.add_argument("--m", type=int, default=10, help="2^m points (default 10: 1,024 runs)")
    parser.add_argument("--first", type=int, default=None, help="run only the first N points")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1, help="threads per worker")
    parser.add_argument("--n-init", type=int, default=10, help="K-means starts per fit (pipeline: 20)")
    parser.add_argument("--uniform-in-watts", action="store_true",
                        help="powers X1 to X6 uniform in W (the check of the distributions)")
    args = parser.parse_args(argv)

    meta = json.loads((Path(args.bundle) / "meta.json").read_text())
    args.runs.mkdir(parents=True, exist_ok=True)
    wanted = settings(args, meta)
    dpath = args.runs / "design.json"
    if dpath.is_file():
        have = json.loads(dpath.read_text())
        differ = [k for k in wanted if k != "m" and have.get(k) != wanted[k]]
        if differ:
            sys.exit(f"{args.runs} holds runs made with other settings ({', '.join(differ)}): "
                     "use another folder")
        wanted["m"] = max(have.get("m", 0), args.m)
    dpath.write_text(json.dumps(wanted, indent=1))

    U = design.sample(args.m, seed=SAMPLE_SEED)
    points = design.transform(U, uniform_in_watts=args.uniform_in_watts)
    last = len(points) if args.first is None else min(args.first, len(points))
    todo = [i for i in range(last) if not all(p.is_file() for p in run_paths(args.runs, i))]
    print(f"{last - len(todo)} of {last} runs already done; {len(todo)} to run with "
          f"{args.workers} worker(s) of {args.threads} thread(s), n_init = {args.n_init}")
    if not todo:
        return
    t0, times, failed = time.time(), [], []

    def progress(i, seconds, out):
        times.append(seconds)
        done = len(times)
        rate = (time.time() - t0) / done
        left = f"{done} of {len(todo)} done, about {hms(rate * (len(todo) - done))} left"
        if out is None:
            failed.append(i)
            print(f"run {i}: FAILED after {seconds:.0f} s, see run_{i:05d}.error; {left}", flush=True)
            return
        print(f"run {i}: {seconds:.0f} s (Y1 {out['Y1']:.3f}, Y2 {out['Y2']:.3f}, Y3 {out['Y3']:.1f}, "
              f"Y5 {out['Y5']:.3f}, Y6 {out['Y6']}, Y8 {out['Y8']:.3f}); {left}", flush=True)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    if args.workers == 1:
        from threadpoolctl import threadpool_limits
        init_worker(args.bundle, args.n_init)
        with threadpool_limits(args.threads):
            for i in todo:
                progress(*one_run(i, points[i], U[i], args.runs))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker,
                                 initargs=(args.bundle, args.n_init, args.threads)) as pool:
            futures = [pool.submit(one_run, i, points[i], U[i], args.runs) for i in todo]
            for f in as_completed(futures):
                progress(*f.result())
    total = time.time() - t0
    print(f"{len(times)} runs in {hms(total)}: {np.mean(times):.0f} s per run in each worker, "
          f"{total / len(times):.0f} s per run overall with {args.workers} worker(s) of "
          f"{args.threads} thread(s); 1,024 runs would take about {hms(total / len(times) * 1024)}")
    if failed:
        print(f"{len(failed)} run(s) failed ({', '.join(map(str, sorted(failed)))}); run the same "
              "command again to retry them")


if __name__ == "__main__":
    main()
