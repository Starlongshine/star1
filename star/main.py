"""
Entry point for NSGA-II Computing Power Internet Scheduling experiments.

Usage examples:
    python main.py                          # default single experiment
    python main.py --mode scalability
    python main.py --mode sensitivity
    python main.py --mode all
    python main.py --n_tasks 100 --n_gen 300
"""

import argparse
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="NSGA-II Computing Power Internet Scheduler"
    )
    p.add_argument("--mode", choices=["single", "scalability", "sensitivity", "all"],
                   default="single")
    p.add_argument("--n_tasks",   type=int,   default=50)
    p.add_argument("--n_core",    type=int,   default=3)
    p.add_argument("--n_regional",type=int,   default=4)
    p.add_argument("--n_edge",    type=int,   default=5)
    p.add_argument("--pop_size",  type=int,   default=100)
    p.add_argument("--n_gen",     type=int,   default=200)
    p.add_argument("--alpha",     type=float, default=0.5,
                   help="Balance coefficient for direction-guided crowding distance")
    p.add_argument("--pref_delay",type=float, default=0.5,
                   help="Preference weight for delay objective [0,1]")
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    from experiment import run_scalability, run_sensitivity, run_single_experiment

    pref = np.array([args.pref_delay, 1.0 - args.pref_delay])

    if args.mode in ("single", "all"):
        run_single_experiment(
            n_tasks=args.n_tasks,
            n_core=args.n_core,
            n_regional=args.n_regional,
            n_edge=args.n_edge,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            preference=pref,
            alpha=args.alpha,
            seed=args.seed,
            scenario_name="main_experiment",
        )

    if args.mode in ("scalability", "all"):
        run_scalability(
            task_sizes=[20, 50, 100, 200],
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            seed=args.seed,
        )

    if args.mode in ("sensitivity", "all"):
        run_sensitivity(
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            seed=args.seed,
        )

    print("\nAll experiments finished. Results saved to ./results/")


if __name__ == "__main__":
    main()
