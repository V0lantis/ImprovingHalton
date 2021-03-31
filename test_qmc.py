import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal, assert_,
                           assert_equal, assert_array_almost_equal,
                           assert_array_equal)

# import _qmc as qmc
import _qmc_cy as qmc
from _qmc_cy import QMCEngine, Halton


class QMCEngineTests:
    """Generic tests for QMC engines."""
    qmce = NotImplemented
    can_scramble = NotImplemented
    unscramble_nd = NotImplemented
    scramble_nd = NotImplemented

    scramble = [True, False]
    ids = ["Scrambled", "Unscrambled"]

    def engine(self, scramble: bool, **kwargs) -> QMCEngine:
        seed = np.random.RandomState(123456)
        if self.can_scramble:
            return self.qmce(scramble=scramble, seed=seed, **kwargs)
        else:
            if scramble:
                pytest.skip()
            else:
                return self.qmce(seed=seed, **kwargs)

    def reference(self, scramble: bool) -> np.ndarray:
        return self.scramble_nd if scramble else self.unscramble_nd

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_0dim(self, scramble):
        engine = self.engine(d=0, scramble=scramble)
        sample = engine.random(4)
        assert_array_equal(np.empty((4, 0)), sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_bounds(self, scramble):
        engine = self.engine(d=100, scramble=scramble)
        sample = engine.random(512)
        assert_(np.all(sample >= 0))
        assert_(np.all(sample <= 1))

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_sample(self, scramble):
        ref_sample = self.reference(scramble=scramble)
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(n=len(ref_sample))

        assert_almost_equal(sample, ref_sample, decimal=1)
        assert engine.num_generated == len(ref_sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_continuing(self, scramble):
        ref_sample = self.reference(scramble=scramble)
        engine = self.engine(d=2, scramble=scramble)

        n_half = len(ref_sample) // 2

        _ = engine.random(n=n_half)
        sample = engine.random(n=n_half)
        assert_almost_equal(sample, ref_sample[n_half:], decimal=1)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_reset(self, scramble):
        ref_sample = self.reference(scramble=scramble)
        engine = self.engine(d=2, scramble=scramble)

        _ = engine.random(n=len(ref_sample) // 2)

        engine.reset()
        assert engine.num_generated == 0

        sample = engine.random(n=len(ref_sample))
        assert_almost_equal(sample, ref_sample, decimal=1)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_fast_forward(self, scramble):
        ref_sample = self.reference(scramble=scramble)
        engine = self.engine(d=2, scramble=scramble)

        engine.fast_forward(4)
        sample = engine.random(n=4)

        assert_almost_equal(sample, ref_sample[4:], decimal=1)

        # alternate fast forwarding with sampling
        engine.reset()
        even_draws = []
        for i in range(8):
            if i % 2 == 0:
                even_draws.append(engine.random())
            else:
                engine.fast_forward(1)
        assert_almost_equal(
            ref_sample[[i for i in range(8) if i % 2 == 0]],
            np.concatenate(even_draws),
            decimal=5
        )

    @pytest.mark.parametrize("scramble", [True])
    def test_distribution(self, scramble):
        d = 50
        engine = self.engine(d=d, scramble=scramble)
        sample = engine.random(1024)
        assert_array_almost_equal(
            np.mean(sample, axis=0), np.repeat(0.5, d), decimal=2
        )
        assert_array_almost_equal(
            np.percentile(sample, 25, axis=0), np.repeat(0.25, d), decimal=2
        )
        assert_array_almost_equal(
            np.percentile(sample, 75, axis=0), np.repeat(0.75, d), decimal=2
        )


class TestHalton(QMCEngineTests):
    qmce = qmc.Halton
    can_scramble = True
    # theoretical values known from Van der Corput
    unscramble_nd = np.array([[0, 0], [1 / 2, 1 / 3],
                              [1 / 4, 2 / 3], [3 / 4, 1 / 9],
                              [1 / 8, 4 / 9], [5 / 8, 7 / 9],
                              [3 / 8, 2 / 9], [7 / 8, 5 / 9]])
    # theoretical values unknown: convergence properties checked
    scramble_nd = np.array([[0.34229571, 0.89178423],
                            [0.84229571, 0.07696942],
                            [0.21729571, 0.41030275],
                            [0.71729571, 0.74363609],
                            [0.46729571, 0.18808053],
                            [0.96729571, 0.52141386],
                            [0.06104571, 0.8547472],
                            [0.56104571, 0.29919164]])
