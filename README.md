# Direct method, IPS and doubly robust OPE: a small simulation

This is the simulation from the article [What Should Machine Learning Be Allowed to Learn in Off-Policy Evaluation?] by Somayeh Farhadi.

A logging policy decides whether to give a renewal discount and records the probability of each decision. From those logs we estimate how much a new discount rule would change the renewal rate, using three estimators: the direct method, inverse propensity scoring (IPS), and a cross-fitted doubly robust (DR) estimator. The outcome model is deliberately weak. With the logging probability known, the weak model biases the direct method but not DR, and DR has less than half the RMSE of IPS.

## Running it

You need Python 3.9 or later and NumPy 1.22 or later.

```
pip install -r requirements.txt
python simulation.py
```

The default run is 10,000 replications of 20,000 decisions and takes about a minute on a recent laptop, longer on slower machines. `python simulation.py --reps 1000` is ten times faster and has more Monte Carlo noise. The results below were produced with Python 3.11 and NumPy 2.4.4, and Python 3.9 with NumPy 1.22 gives the same output.

## Setup

- **Customers.** 20% are in their first year. Each customer also has an engagement score drawn uniformly between 0 and 1.
- **Action.** The discount is either given or not.
- **Logging policy.** Gives the discount to first-year customers with probability 0.05 and to established customers with probability 0.50, and records that probability.
- **Outcome.** Renewal, with probability 0.55 + 0.30 × engagement, plus the effect of the discount when it is given.
- **Effect of the discount.** 4 percentage points for established customers. For first-year customers it is 4 + 8 × (1 − engagement) points, which is 8 points on average. Tenure changes the effect of the discount but not the baseline renewal rate, so the bias of the direct method comes only from the difference in effects.
- **Policies.** The comparator A is the logging policy. The candidate B gives the discount to every first-year customer and follows the logging policy for established customers.
- **Target.** Δ = V(B) − V(A) = 0.20 × 0.95 × 8 = 1.52 percentage points.
- **Outcome model.** One average renewal rate per action, ignoring tenure and engagement, fitted with 5-fold cross-fitting.

## Estimators

All three use the difference between the two policies, g(a, H) = π_B(a | H) − π_A(a | H). IPS and DR also use μ(A | H), the logged probability of the action that was received. Each estimate is the average of a per-decision score, where q̂ is the out-of-fold prediction of the outcome model:

```
direct method:    sum_a g(a, H) * q_hat(a)
IPS:              g(A, H) / mu(A | H) * Y
cross-fitted DR:  sum_a g(a, H) * q_hat(a)  +  g(A, H) / mu(A | H) * (Y - q_hat(A))
```

The 95% intervals for IPS and DR are the estimate ± 1.96 × (standard deviation of the scores) / √n. There is no interval for the direct method: almost all of its error here is bias from the outcome model, which an interval based on sampling variation cannot capture.

## Results

10,000 replications of 20,000 decisions, seed 2026:

| Method | Mean estimate (pp) | Monte Carlo bias (pp) | RMSE (pp) | 95% CI coverage |
| --- | --- | --- | --- | --- |
| True effect | 1.520 | n/a | n/a | n/a |
| Direct method | 0.779 | -0.741 | 0.751 | n/a |
| IPS, known logging probability | 1.509 | -0.011 | 1.232 | 94.5% |
| Cross-fitted DR, known logging probability | 1.512 | -0.008 | 0.584 | 94.8% |

The Monte Carlo standard error of each mean estimate is 0.001 pp for the direct method, 0.012 pp for IPS and 0.006 pp for DR, and that of each coverage figure is 0.2 percentage points. The average half-width of the 95% interval is 2.38 pp for IPS and 1.13 pp for DR.

## Reading the results

- The direct method reports about half of the true effect. Almost all discounted customers in the log (97.6%) are established, so the per-action average learns their 4-point effect and applies it to first-year customers.
- IPS and DR are both centered on the truth. Their biases are within two Monte Carlo standard errors of zero.
- DR uses the same weak outcome model to cut the RMSE of IPS by more than half, from 1.23 to 0.58 points, because its weights multiply the residual Y − q̂(A) and not the full outcome.

## Scope

This is one synthetic setting with a known logging probability, independent decisions and an outcome model that ignores the context. It does not cover estimated propensities, repeated customers or flexible outcome models. The article discusses each of these.
