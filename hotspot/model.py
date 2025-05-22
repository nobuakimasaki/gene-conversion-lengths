import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import product

# Marginal distribution of L (only defined for L >= 2)
def pL_geom_2(l, psi, phi):
    C = phi + psi - phi * psi
    if l >= 2:
        return phi * (1 - phi)**(l - 1) * psi**2 / (C**2)
    else:
        return None

# Probability of 2 <= L <= M 
def pL_geom_2_to_M(l, psi, phi, M):
    C = phi + psi - phi * psi
    return psi**2 * ((1 - phi) - (1 - phi)**M) / (C**2)

#######################
# Distribution of L conditioned on 2 <= L <= M 
def pL_geom_2M(l, psi, phi, M):
    num = pL_geom_2(l, psi, phi)
    denom = pL_geom_2_to_M(l, psi, phi, M)
    return num / denom

# Negative log likelihood for the joint distribution of L (conditioned on 2 <= L <= M)
def neg_log_lik(phi, psi_lst, l_lst, M):
    lik_lst = [pL_geom_2M(l_lst[i], psi_lst[i], phi, M) for i in range(len(l_lst))]
    nll = -np.sum(np.log(lik_lst))
    return nll
#######################

#######################
# Distribution of L conditioned on 2 <= L <= M (when N is a sum of two geometric RVs)
def pL_geom_2M_sum(l, psi, phi, M):
    C = phi + psi - phi * psi
    denom = (C * ((3 - M) * phi * (1 - phi) ** (M - 1) - (1 - phi) ** (M - 1) - 2 * phi + 1) + 2 * phi * (1 - (1 - phi) ** (M - 1)))
    num = C * (l - 3) * phi ** 2 * (1 - phi) ** (l - 2) + 2 * phi ** 2 * (1 - phi) ** (l - 2)
    return num / denom

# Negative log likelihood for the joint distribution of L (conditioned on 2 <= L <= M, when N is a sum of two geometric RVs)
def neg_log_lik_sum(phi, psi_lst, l_lst, M):
    lik_lst = [pL_geom_2M_sum(l_lst[i], psi_lst[i], phi, M) for i in range(len(l_lst))]
    nll = -np.sum(np.log(lik_lst))
    return nll
#######################

#######################
# Distribution of L conditioned on 2 <= L <= M (when N is a mixture of two geometric components)
def pL_geom_2M_mixture(l, psi, w1, phi1, phi2, M):
    w2 = 1 - w1
    num = w1 * pL_geom_2(l, psi, phi1) + w2 * pL_geom_2(l, psi, phi2)
    denom = w1 * pL_geom_2_to_M(l, psi, phi1, M) + w2 * pL_geom_2_to_M(l, psi, phi2, M)
    return num / denom

# Negative log likelihood for the joint distribution of L (conditioned on 2 <= L <= M, when N is a mixture of two geometric components)
def neg_log_lik_mixture(par, psi_lst, l_lst, M, w1):
    phi1, phi2 = par

    lik_lst = [pL_geom_2M_mixture(l_lst[i], psi_lst[i], w1, phi1, phi2, M) for i in range(len(l_lst))]
    nll = -np.sum(np.log(lik_lst))
    return nll
#######################

# Function to find MLE for the geometric and sum models
def optim_geom(psi_lst, l_lst, M):
    bounds = [(0.0005, 0.05)]  # Bounds for phi

    res_geom = minimize(
        neg_log_lik,
        x0=[0.005],  # Initial guess
        args=(psi_lst, l_lst, M),
        method='L-BFGS-B',
        bounds=bounds
    )

    res_geom2 = minimize(
        neg_log_lik_sum,
        x0=[0.005],  # Initial guess
        args=(psi_lst, l_lst, M),
        method='L-BFGS-B',
        bounds=bounds
    )

    return res_geom, res_geom2

# Function to find MLE for mixture model. 
# We start the optimization from the four corners of the parameter space, and pick the result with the smallest nll. 
def optim_mixture(w1, psi_lst, l_lst, M):
    # Define the grid of phi values
    phi_vals = [0.0005, 0.1]
    
    # All combinations of phi1 and phi2
    starts = list(product(phi_vals, repeat=2))
    
    bounds = [(0.0005, 0.1), (0.0005, 0.1)]
    best_result = None

    for start in starts:
        res = minimize(neg_log_lik_mixture, x0=start, args=(psi_lst, l_lst, M, w1),
                       bounds=bounds, method='L-BFGS-B')
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    return best_result

# Function to fit all models
def fit_model_M(psi_lst_filt, l_lst_filt, M):

    # Define grid over w1 and run MLE for each value
    w1_grid = np.concatenate([
    np.arange(0.002, 0.01025, 0.00025),  # fine grid: 0.002 to 0.01
    np.arange(0.05, 0.51, 0.05)])

    results = []

    # Fit mixture model for each w1
    for w1 in w1_grid:
        result = optim_mixture(w1, psi_lst_filt, l_lst_filt, M)
        phi1, phi2 = result.x
        mean_val = w1 / phi1 + (1 - w1) / phi2
        nll = result.fun
        aic = 2 * nll + 6  # 3 parameters (phi1, phi2, w1)
        results.append(("mixture", w1, nll, aic, [phi1, phi2], mean_val))

    # Create DataFrame
    results_df = pd.DataFrame(results, columns=["model", "w1", "NLL", "AIC", "phi", "mean"])
    results_df[["phi1", "phi2"]] = pd.DataFrame(results_df["phi"].tolist(), index=results_df.index)
    results_df = results_df.drop(columns=["phi"]).sort_values(by="NLL")

    # Best mixture model (optional print)
    best = results_df.iloc[0]
    print("\nBest result (mixture):")
    print(f"w1={best['w1']:.3f}, NLL={best['NLL']:.4f}, phi1={best['phi1']:.4f}, phi2={best['phi2']:.4f}, mean={best['mean']:.4f}")

    # Fit null and sum models
    res_null, res_sum = optim_geom(psi_lst_filt, l_lst_filt, M)

    # Null model
    phi_null = res_null.x[0]
    nll_null = res_null.fun
    mean_null = 1 / phi_null
    aic_null = 2 * nll_null + 2  # 1 parameter
    null_row = pd.DataFrame([{
        "model": "null",
        "w1": np.nan,
        "NLL": nll_null,
        "AIC": aic_null,
        "mean": mean_null,
        "phi1": phi_null,
        "phi2": np.nan
    }])

    # Sum model
    phi_sum = res_sum.x[0]
    nll_sum = res_sum.fun
    mean_sum = 2 / phi_sum
    aic_sum = 2 * nll_sum + 2  # 1 parameter
    sum_row = pd.DataFrame([{
        "model": "sum",
        "w1": np.nan,
        "NLL": nll_sum,
        "AIC": aic_sum,
        "mean": mean_sum,
        "phi1": phi_sum,
        "phi2": np.nan
    }])

    return results_df, null_row, sum_row

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)

    # Parameters for simulation
    true_phi = 0.01
    true_psi = 0.002
    M = 1500
    n = 1000  # number of tracts

    # Simulate psi_lst around true_psi (some variability)
    psi_lst = np.random.uniform(true_psi * 0.8, true_psi * 1.2, size=n)

    # Simulate L values using geometric distribution truncated at M
    l_lst = []
    for psi in psi_lst:
        while True:
            # Simulate L from the model: Geometric + 1, truncated at M
            L = np.random.geometric(p=true_phi)
            if 2 <= L <= M:
                l_lst.append(L)
                break
    l_lst = np.array(l_lst)

    # Run geometric and sum models
    res_geom, res_geom2 = optim_geom(psi_lst, l_lst, M)
    print("Geometric model:")
    print("  phi_hat =", res_geom.x[0])
    print("  NLL     =", res_geom.fun)

    print("\nSum of geometrics model:")
    print("  phi_hat =", res_geom2.x[0])
    print("  NLL     =", res_geom2.fun)

    # Run mixture model
    w1 = 0.5  # weight of first component
    res_mixture = optim_mixture(w1, psi_lst, l_lst, M)
    print("\nMixture model:")
    print("  phi1_hat =", res_mixture.x[0])
    print("  phi2_hat =", res_mixture.x[1])
    print("  NLL      =", res_mixture.fun)

    print("\nTesting specific values:")
    test_1 = pL_geom_2M(4, 0.01, 0.02, 1500)
    print(test_1)
    test_2 = pL_geom_2M_sum(4, 0.01, 0.02, 1500)
    print(test_2)
    test_3 = pL_geom_2M_mixture(4, 0.01, 0.2, 0.02, 0.03, 1500)
    print(test_3)

    # print("\nRunning full model fitting...")
    # mixture, null_row, sum_row = fit_model_M(psi_lst, l_lst, M)
    # print("\nResults:")
    # print(mixture, null_row, sum_row)

    w1_grid = np.concatenate([
    np.arange(0.002, 0.01025, 0.00025),  # fine grid: 0.002 to 0.01
    np.arange(0.05, 0.51, 0.05)])

    print(w1_grid)