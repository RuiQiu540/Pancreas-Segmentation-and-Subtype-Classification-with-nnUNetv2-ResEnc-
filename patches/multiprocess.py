from pathlib import Path
from multiprocessing import get_context
from preprocess import preprocess   # assume there is such a preprocess() that crops and resamples

# assume dataset-level fingerprint and plans.json have already been generated

def main():
    in_dir = Path("/path_to_raw")
    out_dir = Path("/path_to_preprocessed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. collect all cases
    names = sorted(in_dir.glob("*.nii.gz"))
    jobs = [(str(p), str(out_dir)) for p in names]

    # 2. create a process pool and dispatch all cases to it
    ctx = get_context("spawn")
    num_workers = 8  # can be adjusted
    with ctx.Pool(num_workers) as pool:
        pool.starmap(preprocess, jobs)

if __name__ == "__main__":
    main()

# this multiprocessing step is meant to be used in the sample-level crop/resample stage
# after the dataset-level planning has been done.
