"""Unit tests for dnlssm.utils.random_state."""

from __future__ import annotations

import numpy as np

from dnlssm.utils.random_state import SeedSequence, spawn_generator


class TestSeedSequence:
    def test_same_label_same_seed_reproducible(self) -> None:
        s1 = SeedSequence(42)
        s2 = SeedSequence(42)
        rng1 = s1.spawn("particle_filter")
        rng2 = s2.spawn("particle_filter")
        np.testing.assert_array_equal(rng1.normal(size=10), rng2.normal(size=10))

    def test_different_labels_give_independent_streams(self) -> None:
        s = SeedSequence(42)
        rng_a = s.spawn("component_a")
        rng_b = s.spawn("component_b")
        draws_a = rng_a.normal(size=1000)
        draws_b = rng_b.normal(size=1000)
        assert not np.allclose(draws_a, draws_b)

    def test_different_tags_give_independent_streams(self) -> None:
        s = SeedSequence(42)
        rng1 = s.spawn("multistart", latent_dim=4, start_index=0)
        rng2 = s.spawn("multistart", latent_dim=4, start_index=1)
        assert not np.allclose(rng1.normal(size=100), rng2.normal(size=100))

    def test_different_global_seed_gives_different_stream(self) -> None:
        rng1 = SeedSequence(1).spawn("x")
        rng2 = SeedSequence(2).spawn("x")
        assert not np.allclose(rng1.normal(size=100), rng2.normal(size=100))

    def test_call_order_independent(self) -> None:
        s1 = SeedSequence(7)
        a1 = s1.spawn("a")
        b1 = s1.spawn("b")

        s2 = SeedSequence(7)
        b2 = s2.spawn("b")
        a2 = s2.spawn("a")

        np.testing.assert_array_equal(a1.normal(size=5), a2.normal(size=5))
        np.testing.assert_array_equal(b1.normal(size=5), b2.normal(size=5))

    def test_spawn_seed_int_reproducible(self) -> None:
        s = SeedSequence(99)
        assert s.spawn_seed_int("scipy_optimizer") == s.spawn_seed_int("scipy_optimizer")


class TestSpawnGeneratorConvenience:
    def test_matches_seed_sequence_output(self) -> None:
        rng1 = spawn_generator(5, "label", tag=1)
        rng2 = SeedSequence(5).spawn("label", tag=1)
        np.testing.assert_array_equal(rng1.normal(size=10), rng2.normal(size=10))
