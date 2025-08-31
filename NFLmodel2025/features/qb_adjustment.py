# features/qb_adjustment.py
def compute_qb_adjustment(row, qb_weight):
    return qb_weight * (row['home_538_qb_adj'] - row['away_538_qb_adj'])