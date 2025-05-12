# cct.py
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

def load_data():
    data_path = Path(__file__).resolve().parent / "../data/plant_knowledge.csv"
    data_path = data_path.resolve()
    df = pd.read_csv(data_path)
    df = df.drop(columns=[df.columns[0]])  # Drop the Informant ID column
    return df.values  # or return df if you prefer DataFrame

def run_cct_model(X):
    N, M = X.shape  # N = number of informants, M = number of questions

    with pm.Model() as model:
        # Priors
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)        # competence of each informant
        Z = pm.Bernoulli("Z", p=0.5, shape=M)                     # consensus (true) answer to each item

        # Reshape for broadcasting
        D_reshaped = D[:, None]  # (N, 1)
        Z_reshaped = Z[None, :]  # (1, M)

        # Compute pij for each (i,j) using: pij = Zj * Di + (1 - Zj) * (1 - Di)
        p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)

        # Likelihood
        X_obs = pm.Bernoulli("X_obs", p=p, observed=X)

        # Sample
        trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True, progressbar=True)

    return model, trace

# Example usage (remove or comment out before final submission)
if __name__ == "__main__":
    X = load_data()
    print("Data shape:", X.shape)

    model, trace = run_cct_model(X)

    # Quick check on inference
    print(az.summary(trace, var_names=["D", "Z"]))

import arviz as az
import matplotlib.pyplot as plt

# Plot posterior competence for each informant
az.plot_posterior(trace, var_names=["D"], hdi_prob=0.94)
plt.tight_layout()
plt.show()
# Plot posterior probability that each answer Z_j is 1
az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.94)
plt.tight_layout()
plt.show()
# Extract posterior means
D_means = trace.posterior["D"].mean(dim=["chain", "draw"]).values
most_competent = np.argmax(D_means)
least_competent = np.argmin(D_means)
print(f"Most competent informant: P{most_competent + 1}")
print(f"Least competent informant: P{least_competent + 1}")

# Plot and save competence estimates (D)
az.plot_posterior(trace, var_names=["D"], hdi_prob=0.94)
plt.tight_layout()
plt.savefig("competence_plot.png")

# Plot and save consensus estimates (Z)
az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.94)
plt.tight_layout()
plt.savefig("consensus_plot.png")

# 1. Compute majority vote per question
majority_vote = (X.sum(axis=0) > (X.shape[0] / 2)).astype(int)  # shape (20,)

# 2. Get posterior mean for Z
Z_means = trace.posterior["Z"].mean(dim=["chain", "draw"]).values  # shape (20,)
Z_pred = (Z_means > 0.5).astype(int)

# 3. Print comparison
print("\nMajority vote vs. CCT model consensus:")
for i, (mv, zp) in enumerate(zip(majority_vote, Z_pred)):
    match = "✓" if mv == zp else "✗"
    print(f"Q{i+1:02}: Majority={mv}  CCT={zp}  [{match}]")

