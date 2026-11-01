"""
Runs the entire pipeline in order: scrape -> external factors -> features
-> train -> dashboard. This is the one command to run regularly (e.g. after
each tender day, or weekly).

USAGE
-----
    python run_all.py

If a step fails (e.g. no internet for one API), the script prints the error
and keeps going with the next step where possible, so a single flaky source
doesn't block the whole run.
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    "1_scrape_incremental.py",
    "2_fetch_external_factors.py",
    "3_build_features.py",
    "4_train_model.py",
    "5_generate_dashboard.py",
]


def run(script):
    print("\n" + "=" * 70, flush=True)
    print(f"RUNNING {script}", flush=True)
    print("=" * 70, flush=True)
    script_path = os.path.join(BASE_DIR, script)
    result = subprocess.run([sys.executable, "-u", script_path], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n!! {script} exited with an error (code {result.returncode}). "
              f"Continuing to the next step anyway.", flush=True)


def main():
    for step in STEPS:
        run(step)
    print("\nAll done. Open dashboard.html in your browser to see the prediction.")


if __name__ == "__main__":
    main()
