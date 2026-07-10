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

import subprocess
import sys

STEPS = [
    "1_scrape_incremental.py",
    "2_fetch_external_factors.py",
    "3_build_features.py",
    "4_train_model.py",
    "5_generate_dashboard.py",
]


def run(script):
    print("\n" + "=" * 70)
    print(f"RUNNING {script}")
    print("=" * 70)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n!! {script} exited with an error (code {result.returncode}). "
              f"Continuing to the next step anyway.")


def main():
    for step in STEPS:
        run(step)
    print("\nAll done. Open dashboard.html in your browser to see the prediction.")


if __name__ == "__main__":
    main()
