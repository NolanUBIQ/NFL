# model/elo.py

import math
import pandas as pd

def elo_to_prob(elo_dif, z=400):
    if z <= 0:
        raise ValueError("Z must be greater than 0")
    return 1 / (1 + math.pow(10, -elo_dif / z))

def prob_to_elo(prob, z=400):
    if z <= 0:
        raise ValueError("Z must be greater than 0")
    return -z * math.log10((1 / prob) - 1)

def shift_calc_helper(margin, line, market_line, config, is_home):
    if is_home:
        line = -line
        market_line = -market_line

    k = config["k"]
    resist = config["market_resist_factor"]
    b = config["b"]

    # Adjust k based on how much the model disagrees with market
    if abs(margin - line) < 1 or resist == 0:
        adj_k = k
    elif abs(line - margin) <= abs(market_line - margin):
        adj_k = k
    else:
        adj_k = k * (1 + (abs(market_line - line) / resist))

    pd_diff = abs(margin - line)
    mult = math.log(max(pd_diff, 1) + 1, b)
    shift = adj_k * mult

    if margin - line < 0:
        shift *= -1

    return shift


def calc_weighted_avg(
    shift_array:list
    ) -> float:
    '''
    Calculates a normalized weighted average for a list of shift / weight pairs
    passed

    Parameters:
    * shift_array (list): An array of shift, weight tuples

    Returns:
    * weighted_avg (float)
    '''
    ## structure ##
    product = 0
    weight = 0
    ## cycle through pairs and populate strucutre
    for pair in shift_array:
        ## check that value exists ##
        if not pd.isnull(pair[0]):
            product += pair[0] * pair[1]
            weight += pair[1]
    ## return ##
    return product / weight

def calc_weighted_shift(
    margin_array:list, model_line:float, market_line:float,
    k:(float or int), b:(float or int), market_resist_factor:(float or int),
    is_home:bool
) -> float:
    '''
    Abstraction that calculates the weighted average elo shift provided a list
    of margin/weight tuples, where margin represents the game outcome measure (ie
    MoV, pff, wepa), and the weight represents how much of the overall shift it should
    represent.

    Paramaters:
    * margin_array (list): array of margin/weight tuple pairs
    * model_line (float): the unregressed expectation of the model
    * market_line (float): the expectation of the market
    * k (float or in): model param that effects degree of shift
    * b (float or int): model param that effects certainty in outcome
    * market_resist_factor (float or int): determins how much to adjust back to
    the market when the model is wrong
    * is_home (bool): whether the home team is being adjusted (as opposed to away team)

    Returns:
    * weighted_shift (float): the weighted average shift to adjust the team by
    '''
    ## generate shifts / weight pairs ##
    shift_pairs = []
    for pair in margin_array:
        ## break out result and weight for clarity
        result = pair[0]
        weight = pair[1]
        ## calc shift
        shift = calc_shift(
            result, model_line, market_line, k,
            b,market_resist_factor, is_home
        )
        ## add with weight to shift pairs ##
        shift_pairs.append(
            (shift, weight)
        )
    ## create weighted average from shift pairs ##
    weighted_avg = calc_weighted_avg(shift_pairs)
    ## return ##
    return weighted_avg