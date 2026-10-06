"""Simulation for the article "What Should Machine Learning Be Allowed to Learn
in Off-Policy Evaluation?" by Somayeh Farhadi.

It compares three estimates of the same policy effect from synthetic renewal
logs: the direct method, inverse propensity scoring (IPS), and a cross-fitted
doubly robust (DR) estimate. The logging probabilities are known, and the
outcome model is deliberately weak.

Usage:
    python simulation.py               # 10,000 replications of 20,000 decisions
    python simulation.py --reps 1000   # about ten times faster, noisier
"""

import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N_DECISIONS = 20_000      # decisions per replication
N_REPLICATIONS = 10_000
N_FOLDS = 5               # cross-fitting folds for the outcome model
SEED = 2026

SHARE_FIRST_YEAR = 0.20           # share of customers in their first year
P_DISCOUNT_FIRST_YEAR = 0.05      # logging policy: P(discount | first-year)
P_DISCOUNT_ESTABLISHED = 0.50     # logging policy: P(discount | established)

# P(renew) = BASE_RENEWAL + ENGAGEMENT_SLOPE * engagement + effect * discount
BASE_RENEWAL = 0.55
ENGAGEMENT_SLOPE = 0.30
EFFECT_ESTABLISHED = 0.04         # effect of the discount for established customers
EFFECT_FIRST_YEAR_MIN = 0.04      # first-year effect: 0.04 + 0.08 * (1 - engagement)
EFFECT_FIRST_YEAR_SLOPE = 0.08


def true_effect():
    """Delta = V(B) - V(A) = E[(pi_B(discount | H) - pi_A(discount | H)) * effect(H)].

    The two policies differ only for first-year customers. B gives them the
    discount with probability 1 and A, the logging policy, with probability
    P_DISCOUNT_FIRST_YEAR. Engagement is uniform on [0, 1], so the average
    first-year effect is EFFECT_FIRST_YEAR_MIN + EFFECT_FIRST_YEAR_SLOPE / 2.
    """
    mean_first_year_effect = EFFECT_FIRST_YEAR_MIN + EFFECT_FIRST_YEAR_SLOPE / 2
    return SHARE_FIRST_YEAR * (1 - P_DISCOUNT_FIRST_YEAR) * mean_first_year_effect


def simulate_log(rng, n):
    """Draw one log of n renewal decisions made by the logging policy."""
    first_year = rng.random(n) < SHARE_FIRST_YEAR
    engagement = rng.random(n)

    # The logging policy decides from tenure only and records its probability.
    p_discount = np.where(first_year, P_DISCOUNT_FIRST_YEAR, P_DISCOUNT_ESTABLISHED)
    discount = rng.random(n) < p_discount

    effect = np.where(
        first_year,
        EFFECT_FIRST_YEAR_MIN + EFFECT_FIRST_YEAR_SLOPE * (1 - engagement),
        EFFECT_ESTABLISHED,
    )
    p_renew = BASE_RENEWAL + ENGAGEMENT_SLOPE * engagement + effect * discount
    renewed = (rng.random(n) < p_renew).astype(float)
    return first_year, p_discount, discount, renewed


def crossfit_constant_outcome_model(rng, discount, renewed, n_folds):
    """Out-of-fold predictions from a deliberately weak outcome model.

    q_hat(a) is the average renewal among training rows that received action a.
    It ignores tenure and engagement. Each row is predicted by a model fitted
    on the other folds.
    """
    n = len(renewed)
    fold = rng.permutation(n) % n_folds
    q_discount = np.empty(n)
    q_no_discount = np.empty(n)
    for k in range(n_folds):
        train, test = fold != k, fold == k
        q_discount[test] = renewed[train & discount].mean()
        q_no_discount[test] = renewed[train & ~discount].mean()
    return q_discount, q_no_discount


def one_replication(rng, n, n_folds):
    """Return (estimate, standard error) for each estimator on one simulated log."""
    first_year, p_discount, discount, renewed = simulate_log(rng, n)

    # Comparator A is the logging policy. Candidate B gives the discount to every
    # first-year customer and follows the logging policy for everyone else.
    # g(a, H) = pi_B(a | H) - pi_A(a | H).
    pi_b_discount = np.where(first_year, 1.0, p_discount)
    g_discount = pi_b_discount - p_discount
    g_no_discount = -g_discount

    # Logged probability of the action that was received, and g at that action.
    mu_received = np.where(discount, p_discount, 1 - p_discount)
    g_received = np.where(discount, g_discount, g_no_discount)

    q_discount, q_no_discount = crossfit_constant_outcome_model(rng, discount, renewed, n_folds)
    q_received = np.where(discount, q_discount, q_no_discount)

    direct = g_discount * q_discount + g_no_discount * q_no_discount
    ips = g_received / mu_received * renewed
    dr = direct + g_received / mu_received * (renewed - q_received)

    return {
        name: (scores.mean(), scores.std(ddof=1) / np.sqrt(n))
        for name, scores in [("direct", direct), ("ips", ips), ("dr", dr)]
    }


def main():
    parser = argparse.ArgumentParser(description="Direct method, IPS and doubly robust OPE simulation.")
    parser.add_argument("--reps", type=int, default=N_REPLICATIONS, help="number of replications")
    parser.add_argument("--n", type=int, default=N_DECISIONS, help="decisions per replication")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    truth = true_effect()
    methods = ["direct", "ips", "dr"]
    labels = {
        "direct": "Direct method",
        "ips": "IPS, known logging probability",
        "dr": "Cross-fitted DR, known logging probability",
    }
    estimates = {m: np.empty(args.reps) for m in methods}
    std_errors = {m: np.empty(args.reps) for m in methods}

    for r in range(args.reps):
        for m, (estimate, std_error) in one_replication(rng, args.n, N_FOLDS).items():
            estimates[m][r] = estimate
            std_errors[m][r] = std_error

    # The direct method gets no interval: its error here is mostly bias from the
    # outcome model, which an interval based on sampling variation cannot capture.
    coverage = {
        m: np.mean(np.abs(estimates[m] - truth) <= 1.96 * std_errors[m]) for m in ["ips", "dr"]
    }

    pp = 100  # report in percentage points
    print(f"{args.reps:,} replications of {args.n:,} decisions, seed {args.seed}\n")
    print("| Method | Mean estimate (pp) | Monte Carlo bias (pp) | RMSE (pp) | 95% CI coverage |")
    print("| --- | --- | --- | --- | --- |")
    print(f"| True effect | {truth * pp:.3f} | n/a | n/a | n/a |")
    for m in methods:
        est = estimates[m]
        bias = est.mean() - truth
        rmse = np.sqrt(np.mean((est - truth) ** 2))
        cov = f"{coverage[m]:.1%}" if m in coverage else "n/a"
        print(f"| {labels[m]} | {est.mean() * pp:.3f} | {bias * pp:.3f} | {rmse * pp:.3f} | {cov} |")

    print("\nMonte Carlo standard error of each mean estimate (pp):")
    for m in methods:
        print(f"  {labels[m]}: {estimates[m].std(ddof=1) / np.sqrt(args.reps) * pp:.3f}")
    print("\nMonte Carlo standard error of each coverage figure (percentage points):")
    for m in coverage:
        print(f"  {labels[m]}: {np.sqrt(coverage[m] * (1 - coverage[m]) / args.reps) * 100:.1f}")
    print("\nAverage half-width of the 95% interval (pp):")
    for m in coverage:
        print(f"  {labels[m]}: {1.96 * std_errors[m].mean() * pp:.3f}")


if __name__ == "__main__":
    main()
