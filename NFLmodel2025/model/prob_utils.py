# model/prob_utils.py

import numpy as np
import math
import scipy.stats

home_keys = [-55, -48, -45, -41, -38, -35, -31, -28, -24, -21, -17, -14, -10, -7, -3,
              3, 7, 10, 14, 17, 21, 24, 28, 31, 35, 38, 41, 45, 48, 55]

params = {
    "baseline_sdev": 13.7,
    "baseline_weight": 1,
    "key_scaling_height": 0.961,
    "key_scaling_sdev": 17.503,
    "key_scaling_flatness": 0.602,
    "key_value_sdev": 0.746,
    "opposite_sign_list": 2.0,
    "zero_value": 0.004
}

def calc_prob(spread, result, parameters):
    baseline = scipy.stats.norm(spread, parameters[0]).pdf(result)

    additional = 0
    for key in home_keys:
        if abs(spread - key) <= 50:
            scale = parameters[2] * math.exp(-2 * ((abs(key - spread) / parameters[3]) ** parameters[4]))
            additional += scale * scipy.stats.norm(key, parameters[5]).pdf(result)

    if (result < 0 and spread > 0) or (result > 0 and spread < 0):
        multiplier = 1 + (baseline * parameters[6])
    else:
        multiplier = 1

    if result == 0:
        return parameters[7]
    return (baseline * parameters[1] + additional) * multiplier

def balance_probs(spread, parameters, possible_results):
    probs = np.array([calc_prob(spread, r, parameters) for r in possible_results])
    probs /= probs.sum()  # Normalize

    while True:
        below = probs[possible_results < spread].sum()
        above = probs[possible_results > spread].sum()
        diff = below - above
        if abs(diff) < 1e-6:
            break

        adj = diff / 2
        probs[possible_results < spread] -= adj / len(probs[possible_results < spread])
        probs[possible_results > spread] += adj / len(probs[possible_results > spread])

    return probs

def calculate_win_prob(spread, parameters=None, possible_results=np.arange(-75, 76)):
    if parameters is None:
        parameters = list(params.values())
    probs = balance_probs(spread, parameters, possible_results)
    return probs[possible_results > 0].sum()
