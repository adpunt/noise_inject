"""
Minimal test suite for NoiseInject.

Run with:  pytest tests/
Covers the core injection/calibration/metrics paths plus a regression test for the
`binary_asymmetric` copy-assignment bug (it previously flipped zero labels).
"""

import sys
import warnings

import numpy as np
import pytest

from noiseInject import (
    NoiseInjectorRegression,
    NoiseInjectorClassification,
    CONDITIONS,
    dose_tolerance,
    DoseWarning,
    calibrate_flip_probability,
    calculate_noise_metrics,
    calculate_classification_metrics,
    calculate_uncertainty_metrics,
)


# --- regression injection: dose matching ------------------------------------
#
# The point of every test below is one property: at a given target, every
# condition must deliver the SAME amount of noise. Six superseded strategies
# delivered between 0.49x and 2.00x the same amount at one nominal setting, and
# their entire apparent severity ordering was explained by that. See
# NOISE_DESIGN.md sections 1, 2 and 2a.

DOSE_MATCHED = [c for c, spec in CONDITIONS.items() if spec['strategy'] != 'censoring']
CENSORING = [c for c, spec in CONDITIONS.items() if spec['strategy'] == 'censoring']
# For censoring the LEVEL is the fraction clipped, not a dose.
CENSORING_LEVEL = 0.25

# The shape is a SECOND axis for grouped-shifted, and it has to be, because the
# registry pins every condition to Gaussian. A Gaussian draw has spread 1, so a
# scale applied at the shape's own spread and a scale applied at the solved one
# come out identical, and the dose check above cannot tell them apart. Laplace
# has spread sqrt 2 and Student-t at nu=5 has sqrt(5/3), which separates them:
# on this axis the old code delivered 1.51x and 1.19x what was asked.
GROUPED_SHIFTED_SHAPES = [
    dict(distribution='gaussian'),
    dict(distribution='laplace'),
    dict(distribution='student_t', nu=5.0),
]
DOSE_MATCHED_CASES = (
    [(c, None) for c in DOSE_MATCHED]
    + [('grouped_shifted', shape) for shape in GROUPED_SHIFTED_SHAPES]
)


def _case_id(case):
    condition, shape = case
    if shape is None:
        return condition
    nu = f"_nu{shape['nu']:g}" if 'nu' in shape else ''
    return f"{condition}-{shape['distribution']}{nu}"


def _injector(condition, shape, seed):
    """A registry condition, optionally with its shape swapped for another.

    The registry itself stays Gaussian -- that is what the study runs. This is
    the test reaching past it to the axis the registry does not cover.
    """
    if shape is None:
        return NoiseInjectorRegression.from_condition(condition, random_state=seed)
    spec = {k: v for k, v in CONDITIONS[condition].items()
            if k not in ('strategy', 'distribution', 'nu')}
    return NoiseInjectorRegression(strategy=CONDITIONS[condition]['strategy'],
                                   random_state=seed, **spec, **shape)


def _labels(n=20000, seed=0):
    """A skewed, strictly positive label column, like a real HOMO-LUMO gap."""
    rng = np.random.RandomState(seed)
    return np.abs(rng.normal(6.8, 1.29, n)) + 0.5


def _groups(n=20000, n_groups=800, seed=1):
    return np.random.RandomState(seed).randint(0, n_groups, n)


def test_regression_inject_changes_values_and_preserves_shape():
    y = _labels(2000)
    y_noisy = NoiseInjectorRegression.from_condition('gaussian', random_state=0).inject(y, 1.0)
    assert y_noisy.shape == y.shape
    assert not np.allclose(y_noisy, y)


def test_regression_seed_is_reproducible():
    y = _labels(2000)
    a = NoiseInjectorRegression.from_condition('gaussian', random_state=42).inject(y, 1.0)
    b = NoiseInjectorRegression.from_condition('gaussian', random_state=42).inject(y, 1.0)
    assert np.array_equal(a, b)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        NoiseInjectorRegression(strategy='not_a_strategy')
    with pytest.raises(ValueError):
        NoiseInjectorRegression(distribution='not_a_distribution')
    with pytest.raises(ValueError):
        NoiseInjectorRegression.from_condition('legacy')      # deleted in 1.0.0


@pytest.mark.parametrize('case', DOSE_MATCHED_CASES, ids=_case_id)
def test_dose_is_flat_across_conditions(case):
    """THE check. Every condition delivers the requested amount, not its own.

    The tolerance is DERIVED per condition from its fourth moment and its
    effective number of independent contributions (`dose_tolerance`), not
    hand-kept. A list of "unstable conditions" would need editing every time a
    condition is added, and would silently stop covering the new one.
    """
    condition, shape = case
    y, g = _labels(), _groups()
    tau = 0.5 * y.std()
    runs = [_injector(condition, shape, seed).inject_verbose(y, tau, groups=g)
            for seed in range(20)]

    # 1. The solved scale hits the target EXACTLY -- this part is arithmetic,
    #    and it is the half of the gate that does not depend on a draw.
    for r in runs:
        assert r.unit_dose_g * r.solved_scale == pytest.approx(tau, rel=1e-12)

    # 2. The MEAN realised dose hits it. A single realisation cannot: at
    #    n = 20,000 the sampling spread of an RMS estimate is already 0.5%, and
    #    for grouped-shifted the group term has only a few hundred degrees of
    #    freedom. Section 2a rule 3 -- fix the population dose, record the
    #    realised one.
    mean_dose = float(np.mean([r.realised_dose_label_units for r in runs]))
    r0 = runs[0]
    tol = dose_tolerance(r0.epsilon, r0.effective_n,
                         nu=(shape or CONDITIONS[condition]).get('nu')) / np.sqrt(len(runs))
    assert mean_dose == pytest.approx(tau, rel=max(tol, 0.002)), (
        f"{_case_id(case)}: asked for {tau:.4f}, delivered {mean_dose:.4f} on average"
        f" (tolerance {100 * tol:.2f}%, effective n {r0.effective_n:.0f})")


def test_the_delivered_dose_is_checked_not_just_recorded():
    """The injector must SAY SO when it hands back an amount it was not asked for.

    Every field of the provenance was written and nothing ever read it back, so
    grouped-shifted delivered up to 1.51x its target under a heavy tail for the
    life of the project without a single check going red.

    It warns rather than raising. The band is three standard errors, so a working
    injector trips it by chance on about 1% of draws at the sizes the
    experimental datasets have, and stopping there would discard a sound run over
    an unlucky draw. What must not happen is silence.

    The draw is scaled behind the injector's back, which is the only way to
    produce a wrong amount once the code is right -- the point is that the check
    exists and fires, not that any condition still miscalibrates.
    """
    y, g = _labels(5000), _groups(5000)
    for condition in DOSE_MATCHED:
        inj = NoiseInjectorRegression.from_condition(condition, random_state=0)
        honest = inj._draw_shape
        inj._draw_shape = lambda n, _f=honest, **kw: 2.0 * _f(n, **kw)
        with pytest.warns(DoseWarning, match='outside the'):
            r = inj.inject_verbose(y, 0.4, groups=g)
        # And it carries on, with the amount it really delivered on the row.
        assert r.as_row()['realised_dose_label_units'] > 0.6, condition


def test_a_draw_that_lands_inside_the_band_is_silent():
    """No warning on an ordinary draw, or the real one is lost in the noise."""
    y, g = _labels(), _groups()
    with warnings.catch_warnings():
        warnings.simplefilter('error', DoseWarning)
        for condition in DOSE_MATCHED:
            NoiseInjectorRegression.from_condition(
                condition, random_state=0).inject_verbose(y, 0.5 * y.std(), groups=g)


def test_censoring_is_exempt_from_the_dose_check():
    """Censoring is swept on the fraction clipped, not on an amount, so there
    is no target for a delivered amount to be compared against."""
    y, g = _labels(5000), _groups(5000)
    r = NoiseInjectorRegression.from_condition('censoring', random_state=0).inject_verbose(
        y, CENSORING_LEVEL, groups=g)
    assert np.isnan(r.target_dose_label_units)
    assert r.realised_dose_label_units > 0


def test_zero_dose_records_exactly_zero():
    """Not a small number -- zero. The negative control the old code never had.

    The previous pipeline reconstructed the injected noise by regressing the
    noisy label on the clean one; at zero noise the residuals were
    floating-point rounding, whose size grows with the label -- which is exactly
    where uncertainty is largest. The zero-noise control therefore showed a
    STRONGER signal than the real levels did.
    """
    y, g = _labels(5000), _groups(5000)
    for condition in DOSE_MATCHED:
        r = NoiseInjectorRegression.from_condition(condition, random_state=0).inject_verbose(
            y, 0.0, groups=g)
        assert np.array_equal(r.epsilon, np.zeros(len(y))), condition
        assert np.array_equal(r.y_noisy, y), condition


@pytest.mark.parametrize('condition', DOSE_MATCHED + CENSORING)
def test_recorded_noise_reconstructs_the_label_exactly(condition):
    """y_clean + epsilon == y_noisy, bit for bit. Recorded, never reconstructed."""
    y, g = _labels(5000), _groups(5000)
    level = CENSORING_LEVEL if condition in CENSORING else 0.4
    r = NoiseInjectorRegression.from_condition(condition, random_state=3).inject_verbose(
        y, level, groups=g)
    assert np.array_equal(r.y_clean + r.epsilon, r.y_noisy)


def test_student_t_reduces_to_gaussian_in_the_limit():
    """Gaussian is Student-t's nu -> infinity limit, so the two must nest."""
    y = _labels()
    tau = 0.5 * y.std()
    heavy = NoiseInjectorRegression(strategy='uniform', distribution='student_t',
                                    nu=200.0, random_state=11).inject_verbose(y, tau)
    normal = NoiseInjectorRegression.from_condition('gaussian',
                                                    random_state=11).inject_verbose(y, tau)
    assert heavy.unit_dose_g == pytest.approx(normal.unit_dose_g, abs=0.006)
    frac = lambda r: np.mean(np.abs(r.epsilon) > 3 * tau)
    assert frac(heavy) == pytest.approx(frac(normal), abs=0.002)


def test_student_t_rejects_undefined_variance():
    """At nu <= 2 the variance is undefined and dose matching stops meaning anything."""
    for nu in (2.0, 1.5, 0.5):
        with pytest.raises(ValueError):
            NoiseInjectorRegression(strategy='uniform', distribution='student_t', nu=nu)
    with pytest.raises(ValueError):
        NoiseInjectorRegression(strategy='uniform', distribution='student_t')


def test_conditions_are_distinguishable_at_matched_dose():
    """Matched amount must not mean matched shape, or there is nothing to compare.

    NOISE_DESIGN.md section 5.2 measured an eight-fold spread in how many labels
    end up badly wrong at identical total noise.
    """
    y, g = _labels(), _groups()
    tau = 0.5 * y.std()
    frac = {}
    for condition in ('gaussian', 'student_t_nu5', 'student_t_nu3', 'grouped_wider',
                      'outlier_p10'):
        r = NoiseInjectorRegression.from_condition(condition, random_state=5).inject_verbose(
            y, tau, groups=g)
        frac[condition] = float(np.mean(np.abs(r.epsilon) > 3 * tau))
    assert frac['gaussian'] < 0.005
    assert frac['student_t_nu3'] > 3 * frac['gaussian']
    assert frac['grouped_wider'] > 3 * frac['gaussian']


# --- the two grouped conditions ---------------------------------------------

def test_grouped_requires_a_group_assignment():
    y = _labels(1000)
    for condition in ('grouped_wider', 'grouped_shifted'):
        with pytest.raises(ValueError):
            NoiseInjectorRegression.from_condition(condition, random_state=0).inject_verbose(y, 0.3)


def test_grouped_wider_records_the_realised_molecule_fraction():
    """Selection is by molecule fraction, not group fraction (section 2a rule 1).

    Real Murcko scaffolds are very unevenly sized: on QM9 a nominal group
    fraction of 0.2 delivered an affected molecule fraction anywhere between
    0.067 and 0.551. Whatever is realised must be recorded, because the
    solver divides by it.
    """
    y = _labels()
    # Deliberately lopsided groups, like real scaffolds: one holds a third.
    groups = np.concatenate([np.zeros(len(y) // 3, dtype=int),
                             np.arange(len(y) - len(y) // 3) + 1])
    r = NoiseInjectorRegression.from_condition('grouped_wider', random_state=0).inject_verbose(
        y, 0.5, groups=groups)
    assert 0.15 <= r.affected_molecule_fraction <= 0.40
    assert r.n_groups == len(np.unique(groups))
    assert r.largest_group_share == pytest.approx(1 / 3, abs=0.01)
    # Only the affected molecules get the wider scale, and it is lambda times wider.
    scales = np.unique(np.round(r.noise_scale, 10))
    assert len(scales) == 2
    assert scales[1] / scales[0] == pytest.approx(3.0, rel=1e-6)


def test_held_out_molecules_are_scored_against_the_TRAINING_selection():
    """The pattern a held-out molecule is scored against must describe the
    injection that actually happened.

    `noise_scale` re-runs the group selection on whatever group array it is
    given. Without `reference_groups` that means the held-out molecules' own
    groups, which picks a DIFFERENT set: on a 40-group split, two of the eight
    groups corrupted in training went unmarked. Question B -- does the model
    learn where the data is unreliable -- would then be scored against an
    injection that never happened, and for the grouped conditions that is the
    whole question.
    """
    rng = np.random.RandomState(0)
    g_train, g_test = rng.randint(0, 40, 800), rng.randint(0, 40, 200)
    y_train, y_test = _labels(800, seed=0), _labels(200, seed=5)

    inj = NoiseInjectorRegression.from_condition('grouped_wider', random_state=7)
    injected = inj.inject_verbose(y_train, 0.5, groups=g_train)
    hit = {int(g) for g in np.unique(g_train[injected.noise_scale >
                                             injected.noise_scale.min()])}
    assert len(hit) > 1

    right = inj.noise_scale(y_test, 0.5, reference=y_train, groups=g_test,
                            reference_groups=g_train)
    marked = {int(g) for g in np.unique(g_test[right > right.min()])}
    assert marked == hit & set(g_test.tolist()), (
        f"scored against {sorted(marked)}, training was corrupted in {sorted(hit)}")

    # And the bug is real: without reference_groups the two sets differ.
    wrong = inj.noise_scale(y_test, 0.5, reference=y_train, groups=g_test)
    assert {int(g) for g in np.unique(g_test[wrong > wrong.min()])} != marked


def test_grouped_shifted_declares_its_scale_degenerate():
    """Every group's offset is drawn from the same distribution, so every
    molecule is equally affected and the per-molecule scale carries no
    information. What differs by group is the DIRECTION -- the mechanism -- not
    a magnitude a model could rank. The flag must agree with the array."""
    y, g = _labels(3000), _groups(3000)
    inj = NoiseInjectorRegression.from_condition('grouped_shifted', random_state=0)
    assert inj.scale_is_degenerate
    assert len(np.unique(inj.noise_scale(y, 0.4, groups=g))) == 1
    assert inj.inject_verbose(y, 0.4, groups=g).scale_is_degenerate


def test_grouped_shifted_moves_whole_groups_together():
    """Every molecule in a group shares one offset -- that is the mechanism.

    Not zero-mean in any one run, and the offsets are deliberately not centred.
    """
    y = _labels(6000)
    groups = np.repeat(np.arange(60), 100)
    r = NoiseInjectorRegression.from_condition('grouped_shifted', random_state=2).inject_verbose(
        y, 0.5, groups=groups)
    per_group = np.array([r.epsilon[groups == g].mean() for g in range(60)])
    flat = NoiseInjectorRegression.from_condition('gaussian', random_state=2).inject_verbose(
        y, 0.5)
    per_group_flat = np.array([flat.epsilon[groups == g].mean() for g in range(60)])
    # rho = 0.62 of the variance sits between groups, so group means must be
    # far more spread out than they would be under any ungrouped condition.
    assert per_group.std() > 5 * per_group_flat.std()
    assert per_group.std() == pytest.approx(np.sqrt(0.62) * 0.5, rel=0.25)


# --- censoring ---------------------------------------------------------------

def test_only_censoring_and_grouped_shifted_move_labels_off_centre():
    """Scatter versus bias -- the comparison the whole study now turns on.

    Models can average away unbiased scatter but not a systematic shift, which
    is why censoring costs twelve times more than any difference between the
    zero-mean shapes. Grouped-shifted is the second, independent demonstration
    of the same effect at the level of a chemical family, so it is biased per
    run BY DESIGN and does not belong with the scatter conditions.
    """
    y, g = _labels(), _groups()
    tau = 0.5 * y.std()
    zero_mean = [c for c in DOSE_MATCHED if c != 'grouped_shifted']
    scatter = [abs(NoiseInjectorRegression.from_condition(c, random_state=4).inject_verbose(
        y, tau, groups=g).mean_epsilon) for c in zero_mean]
    shifted = NoiseInjectorRegression.from_condition(
        'grouped_shifted', random_state=4).inject_verbose(y, tau, groups=g)
    cens = NoiseInjectorRegression.from_condition('censoring', random_state=4).inject_verbose(
        y, 0.25, groups=g)

    assert max(scatter) < 0.02 * tau                 # the five scatter conditions stay centred
    assert abs(shifted.mean_epsilon) > 5 * max(scatter)
    assert abs(cens.mean_epsilon) > 10 * max(scatter)
    assert cens.mean_epsilon < 0                       # an upper limit lowers labels


def test_censoring_clips_the_requested_fraction_and_nothing_else():
    y = _labels()
    r = NoiseInjectorRegression.from_condition('censoring', random_state=0).inject_verbose(y, 0.30)
    assert r.affected_molecule_fraction == pytest.approx(0.30, abs=0.005)
    assert r.as_row()['noise_type'] == 'censoring_30'      # the name is derived
    assert np.all(r.y_noisy <= r.censoring_limit + 1e-9)
    untouched = r.epsilon == 0
    assert np.all(r.y_noisy[untouched] == y[untouched])
    assert np.isnan(r.unit_dose_g) and np.isnan(r.target_dose_label_units)   # not dose-matched


def test_censoring_scores_held_out_molecules_against_the_training_limit():
    """The cut-point comes from the TRAINING labels, or held-out molecules are
    scored against a distribution they were never exposed to."""
    y_train, y_test = _labels(8000, seed=0), _labels(4000, seed=99) + 2.0
    inj = NoiseInjectorRegression.from_condition('censoring', random_state=0)
    scale_own = inj.noise_scale(y_test, 0.20)
    scale_ref = inj.noise_scale(y_test, 0.20, reference=y_train)
    assert not np.allclose(scale_own, scale_ref)
    assert (scale_ref > 0).mean() > (scale_own > 0).mean()


# --- the provenance every row must carry -------------------------------------

def test_every_provenance_field_is_populated():
    """A blank provenance column is how the dose confound survived for years."""
    y, g = _labels(4000), _groups(4000)
    for condition in DOSE_MATCHED + CENSORING:
        level = CENSORING_LEVEL if condition in CENSORING else 0.4
        row = NoiseInjectorRegression.from_condition(condition, random_state=8).inject_verbose(
            y, level, groups=g).as_row()
        for key in ('noise_type', 'shape_name', 'targeting_name',
                    'realised_dose_label_units', 'realised_dose_fraction_of_spread',
                    'affected_molecule_fraction', 'mean_epsilon', 'effective_n',
                    'n_groups', 'largest_group_share', 'clean_label_sd', 'seed'):
            assert row[key] is not None, f"{condition}: {key} is blank"
        expected = 'censoring_25' if condition in CENSORING else condition
        assert row['noise_type'] == expected


def test_shape_only_conditions_declare_their_scale_degenerate():
    """For Gaussian/Student-t/Laplace every molecule gets the same scale, so
    "which region is unreliable" is UNDEFINED, not zero. Say so, loudly."""
    y, g = _labels(3000), _groups(3000)
    flat = NoiseInjectorRegression.from_condition('gaussian', random_state=0).inject_verbose(y, 0.4)
    assert flat.scale_is_degenerate
    assert len(np.unique(flat.noise_scale)) == 1
    structured = NoiseInjectorRegression.from_condition(
        'grouped_wider', random_state=0).inject_verbose(y, 0.4, groups=g)
    assert not structured.scale_is_degenerate
    assert len(np.unique(structured.noise_scale)) > 1


def test_effective_noise_measures_the_second_moment():
    """The default was mean |dy|, the first moment. Matching it hands the
    heavy-tailed conditions up to 24% more actual noise at one setting."""
    y = _labels()
    inj = NoiseInjectorRegression.from_condition('gaussian', random_state=0)
    r = inj.inject_verbose(y, 0.4)
    assert inj.get_effective_noise(y, r.y_noisy, method='rms') == pytest.approx(0.4, rel=0.01)
    assert inj.get_effective_noise(y, r.y_noisy) == pytest.approx(0.4 / y.std(), rel=0.01)


# --- classification injection ----------------------------------------------

def test_uniform_flips_some_labels():
    y = np.array([0, 1, 2] * 100)
    y_noisy = NoiseInjectorClassification('uniform', random_state=0).inject(y, 0.3)
    assert 0.0 < np.mean(y != y_noisy) < 1.0


def test_binary_asymmetric_actually_flips():
    """Regression test: binary_asymmetric used to assign into a copy and flip nothing."""
    y = np.array([0, 1] * 500)
    inj = NoiseInjectorClassification('binary_asymmetric', random_state=0)
    y_noisy = inj.inject(y, 0.2, flip_01_mult=1.5, flip_10_mult=0.5)
    overall = np.mean(y != y_noisy)
    flip_0to1 = np.mean(y_noisy[y == 0] != 0)
    flip_1to0 = np.mean(y_noisy[y == 1] != 1)
    assert overall > 0.0                      # the bug made this exactly 0
    assert flip_0to1 > flip_1to0              # asymmetry holds (0->1 rate > 1->0 rate)


# --- calibration ------------------------------------------------------------
# Regression has none: every condition solves for its own scale in closed form.
# `calibrate_sigma` was deleted in 1.0.0 -- it searched on the FIRST moment.

def test_calibrate_flip_probability_hits_target_flip_rate():
    y = np.array([0, 1] * 500)
    p = calibrate_flip_probability(y, target_flip_rate=0.2, random_state=0)
    inj = NoiseInjectorClassification('uniform', random_state=0)
    assert abs(inj.get_effective_flip_rate(y, inj.inject(y, p)) - 0.2) < 0.05


# --- metrics ----------------------------------------------------------------

def test_regression_metrics_return_expected_columns():
    rng = np.random.RandomState(0)
    y_true = rng.normal(0, 1, 100)
    predictions = {0.0: y_true.copy(), 1.0: y_true + rng.normal(0, 0.5, 100)}
    per_sigma, summary = calculate_noise_metrics(y_true, predictions)
    assert 'r2' in per_sigma.columns
    # auc_norm / Weibull replace the old NSI slope; Weibull column is present even with
    # <4 sigma points (value is NaN, see the >=4-point test below).
    assert 'auc_norm_r2' in summary.columns
    assert 'weibull_beta_r2' in summary.columns
    assert 'curve_stable_r2' in summary.columns
    assert 'retention_pct_r2' in summary.columns
    assert not any(c.startswith('nsi_') for c in summary.columns)


def test_classification_metrics_return_three_frames():
    rng = np.random.RandomState(0)
    y_true = np.array([0, 1, 2] * 30)
    predictions = {0.0: y_true.copy(), 0.2: rng.permutation(y_true)}
    per_flip, summary, per_class = calculate_classification_metrics(y_true, predictions)
    assert 'accuracy' in per_flip.columns
    assert 'auc_norm_accuracy' in summary.columns
    assert 'auc_norm_f1_class_0' in summary.columns
    assert 'class' in per_class.columns
    assert not any(c.startswith('nsi_') for c in summary.columns)


def test_auc_norm_flat_curve_is_one_and_stable():
    """A perfectly flat R2 curve retains 100% of skill: auc_norm == 1, curve marked stable."""
    y_true = np.linspace(-3, 3, 200)
    predictions = {s: y_true + 0.5 * np.sin(y_true) for s in (0.0, 0.25, 0.5, 0.75, 1.0)}
    _, summary = calculate_noise_metrics(y_true, predictions)
    assert abs(summary['auc_norm_r2'].iloc[0] - 1.0) < 1e-9
    assert summary['curve_stable_r2'].iloc[0] is True or summary['curve_stable_r2'].iloc[0] == True  # noqa: E712


def test_auc_norm_matches_reference_formula():
    """auc_norm / weibull_beta equal the standalone reference helpers on a hand-built curve."""
    from noiseInject.metrics import _retention_auc_norm, _retention_weibull

    sig = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    r2 = np.array([0.90, 0.88, 0.82, 0.70, 0.50])
    # Build predictions that reproduce exactly this r2 curve on a fixed y_true.
    rng = np.random.RandomState(7)
    y_true = rng.normal(0, 1, 4000)
    var = np.var(y_true)
    predictions = {}
    for s, target in zip(sig, r2):
        # r2 = 1 - MSE/var  ->  MSE = (1-r2)*var ; add zero-mean noise of that variance
        noise = rng.normal(0, np.sqrt((1 - target) * var), y_true.size)
        predictions[float(s)] = y_true + noise
    per_sigma, summary = calculate_noise_metrics(y_true, predictions)
    obs_r2 = per_sigma.sort_values('sigma')['r2'].values
    base = obs_r2[0]
    exp_auc = _retention_auc_norm(sig, obs_r2, base)
    exp_tau, exp_beta = _retention_weibull(sig, obs_r2, base)
    assert abs(summary['auc_norm_r2'].iloc[0] - exp_auc) < 1e-9
    assert abs(summary['weibull_beta_r2'].iloc[0] - exp_beta) < 1e-6


def test_curve_stable_false_on_negative_retention():
    """A retrain that collapses to negative R2 leaves auc_norm finite but flags instability."""
    rng = np.random.RandomState(1)
    y_true = rng.normal(0, 1, 300)
    predictions = {
        0.0: y_true.copy(),
        0.25: y_true + rng.normal(0, 0.3, 300),
        0.5: y_true[::-1] * 5.0,          # anti-correlated blow-up -> R2 << 0
        0.75: y_true + rng.normal(0, 0.6, 300),
        1.0: y_true + rng.normal(0, 0.8, 300),
    }
    _, summary = calculate_noise_metrics(y_true, predictions)
    assert summary['curve_stable_r2'].iloc[0] == False  # noqa: E712
    assert np.isfinite(summary['auc_norm_r2'].iloc[0])   # value still reported, not masked


def test_baseline_threshold_gates_weak_models():
    """With baseline_threshold set above the clean R2, curve scalars are NaN; default keeps them."""
    rng = np.random.RandomState(2)
    y_true = rng.normal(0, 1, 300)
    predictions = {s: y_true + rng.normal(0, 0.4 + s, 300) for s in (0.0, 0.25, 0.5, 0.75, 1.0)}

    _, summary_default = calculate_noise_metrics(y_true, predictions)
    assert np.isfinite(summary_default['auc_norm_r2'].iloc[0])  # default: no gating

    _, summary_gated = calculate_noise_metrics(y_true, predictions, baseline_threshold=0.99)
    assert np.isnan(summary_gated['auc_norm_r2'].iloc[0])
    assert np.isnan(summary_gated['weibull_beta_r2'].iloc[0])


# --- uncertainty metrics ----------------------------------------------------

def test_uncertainty_metrics_return_expected_columns():
    rng = np.random.RandomState(0)
    y_true = rng.normal(0, 1, 500)
    predictions, uncertainties = {}, {}
    for sigma in (0.0, 0.5):
        err = rng.normal(0, 0.2 + sigma, 500)
        predictions[sigma] = y_true + err
        uncertainties[sigma] = np.full(500, 0.2 + sigma)
    per_sigma, summary = calculate_uncertainty_metrics(y_true, predictions, uncertainties)
    for col in ('unc_error_rho', 'ece', 'coverage_1sigma', 'coverage_2sigma',
                'mean_interval_width'):
        assert col in per_sigma.columns
    assert 'baseline_ece' in summary.columns
    assert 'miscoverage_1sigma' in summary.columns


def test_uncertainty_error_rho_high_when_u_tracks_error():
    """If predicted uncertainty equals the absolute error, the rank correlation is ~1."""
    rng = np.random.RandomState(1)
    y_true = rng.normal(0, 1, 300)
    err = rng.uniform(0, 2, 300)
    y_pred = y_true + err
    predictions = {0.0: y_pred}
    uncertainties = {0.0: np.abs(err)}  # u == |error| exactly
    per_sigma, _ = calculate_uncertainty_metrics(y_true, predictions, uncertainties)
    assert per_sigma['unc_error_rho'].iloc[0] > 0.99


def test_coverage_matches_gaussian_targets_when_well_calibrated():
    """Perfectly-calibrated Gaussian errors should hit ~68% / ~95% coverage."""
    rng = np.random.RandomState(2)
    n = 20000
    y_true = np.zeros(n)
    true_sigma = 0.5
    y_pred = y_true + rng.normal(0, true_sigma, n)
    predictions = {0.0: y_pred}
    uncertainties = {0.0: np.full(n, true_sigma)}
    per_sigma, _ = calculate_uncertainty_metrics(y_true, predictions, uncertainties)
    assert abs(per_sigma['coverage_1sigma'].iloc[0] - 0.6827) < 0.02
    assert abs(per_sigma['coverage_2sigma'].iloc[0] - 0.9545) < 0.02


def test_uncertainty_noise_rho_appears_only_with_injected_noise():
    rng = np.random.RandomState(3)
    y_true = rng.normal(0, 1, 400)
    predictions = {0.0: y_true.copy(), 0.5: y_true + rng.normal(0, 0.5, 400)}
    uncertainties = {0.0: np.full(400, 0.1), 0.5: np.full(400, 0.5)}

    per_sigma, _ = calculate_uncertainty_metrics(y_true, predictions, uncertainties)
    assert 'unc_noise_rho' not in per_sigma.columns

    # Inject noise whose magnitude is mirrored by the predicted uncertainty.
    eps = {s: rng.uniform(0, 1, 400) for s in (0.0, 0.5)}
    noise_unc = {s: eps[s].copy() for s in (0.0, 0.5)}
    per_sigma2, _ = calculate_uncertainty_metrics(
        y_true, predictions, uncertainties,
        injected_noise=eps, noise_uncertainties=noise_unc,
    )
    assert 'unc_noise_rho' in per_sigma2.columns
    assert per_sigma2['unc_noise_rho'].iloc[0] > 0.99


def test_split_conformal_achieves_target_coverage():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from noiseInject.wrappers import SplitConformalRegressor

    X, y = make_regression(n_samples=900, n_features=8, noise=10.0, random_state=0)
    X_tr, X_cal, X_te = X[:500], X[500:700], X[700:]
    y_tr, y_cal, y_te = y[:500], y[500:700], y[700:]

    cp = SplitConformalRegressor(RandomForestRegressor(random_state=0), coverage=0.9)
    cp.fit(X_tr, y_tr, X_cal, y_cal)
    lo, hi = cp.predict_interval(X_te)
    coverage = np.mean((y_te >= lo) & (y_te <= hi))
    assert coverage > 0.8  # finite-sample, but should be near the 0.9 target


def test_mc_dropout_wrapper_returns_mean_and_std():
    torch = pytest.importorskip("torch")  # requires noiseInject[uncertainty]
    import torch.nn as nn
    from noiseInject.wrappers import MCDropoutRegressor

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))
    X = np.random.RandomState(0).normal(size=(40, 5))
    mean, std = MCDropoutRegressor(net, n_forward=50).predict(X)
    assert mean.shape == (40,) and std.shape == (40,)
    assert np.all(std >= 0) and std.mean() > 0  # dropout induces non-zero spread


def test_gauche_gp_wrapper_rbf_fits_and_predicts():
    pytest.importorskip("gpytorch")  # requires noiseInject[uncertainty]
    from noiseInject.wrappers import GaucheGPRegressor

    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.1, 60)
    gp = GaucheGPRegressor(kernel='rbf', training_iter=30).fit(X, y)
    mean, std = gp.predict(X[:10])
    assert mean.shape == (10,) and std.shape == (10,)
    assert np.all(std > 0)


if __name__ == '__main__':
    # Runnable as a plain script so `scripts/check_fixes_fail_when_removed.py`
    # in qsar_qm_models can point at a path and pass `-k` through.
    raise SystemExit(pytest.main([__file__, '-q'] + sys.argv[1:]))
