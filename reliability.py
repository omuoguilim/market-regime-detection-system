"""Transparent diagnostics, not a validated fragility score."""
from itertools import permutations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_profiles(previous, current, scale):
    """One-to-one mean-profile matching. Scale must be determined from past data.

    Return current indices in previous order, distances, and assignment gap.
    Margin is the second-best minus best total cost; it is NOT a probability.
    Means only: this does not establish persistence of entire distributions.
    """
    a, b, s = np.asarray(previous), np.asarray(current), np.asarray(scale)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 3:
        raise ValueError('This implementation requires three equal-size state profiles.')
    if not all(np.isfinite(x).all() for x in (a,b,s)) or (s <= 0).any():
        raise ValueError('Finite means and positive scale required.')
    costs = np.linalg.norm((a[:,None,:]-b[None,:,:])/s, axis=2)
    rows, cols = linear_sum_assignment(costs)
    alternatives = sorted(sum(costs[i,j] for i,j in enumerate(p)) for p in permutations(range(3)))
    return cols, costs[rows,cols], alternatives[1]-alternatives[0]


def ensemble_diagnostics(probs):
    """Input (accepted seeds, dates, aligned states). All seeds weighted equally."""
    p = np.asarray(probs, dtype=float)
    if p.ndim != 3 or p.shape[0] < 1 or p.shape[2] != 3:
        raise ValueError('Expected seed/date/three-state probabilities.')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=2),1):
        raise ValueError('Invalid probabilities.')
    n, t, _ = p.shape
    if n < 2:
        return {key: np.full(t,np.nan) for key in
                ['vote_disagreement','js_disagreement','exposure_std']}
    def entropy(x):
        return -(x*np.log(np.clip(x,1e-300,1))).sum(axis=-1)
    votes = p.argmax(axis=2)
    counts = np.stack([(votes==i).sum(axis=0) for i in range(3)])
    js = (entropy(p.mean(axis=0))-entropy(p).mean(axis=0))/np.log(3)
    return dict(vote_disagreement=1-counts.max(axis=0)/n,
                js_disagreement=np.maximum(0,js),
                exposure_std=(p[:,:,2]+.5*p[:,:,1]).std(axis=0))


def lineage(folds, scale):
    """Match each new fit to preceding profiles in stable lineage order."""
    if not folds:
        raise ValueError('No folds supplied.')
    prior = None
    rows = []
    for fold in folds:
        means = np.asarray(fold['raw_means'])
        semantic = {state:label for state,label in zip(fold['state_order'],['Bear','Transition','Bull'])}
        if prior is None:
            mapping = np.asarray(fold['state_order'])
            distances = np.zeros(3)
            margin = np.nan
        else:
            mapping, distances, margin = match_profiles(prior, means, scale)
        for identity, (state, distance) in enumerate(zip(mapping, distances)):
            rows.append(dict(fold=fold['fold'], available_from=fold['test_start'],
                lineage=f'L{identity}', initially_named=['Bear','Transition','Bull'][identity],
                current_label=semantic[int(state)], state_number=int(state),
                profile_distance=float(distance), assignment_margin=float(margin),
                mean_return=means[state,0], mean_volatility=means[state,1],
                mean_momentum=means[state,2], mean_volume_change=means[state,3]))
        prior = means[mapping]
    return pd.DataFrame(rows)
