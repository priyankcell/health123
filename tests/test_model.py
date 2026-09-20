import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twincare_glyco.features import BaselineProfile, DynamicState
from twincare_glyco.model import predict_spike_probability, train_baseline_model


class TestTwinCareModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = train_baseline_model(seed=7)
        self.profile = BaselineProfile(
            age=58,
            diabetes=1,
            hypertension=1,
            hba1c=8.2,
            bmi=29.7,
        )

    def test_probability_is_valid(self) -> None:
        state = DynamicState(158, 18, 30, 92, 31, 4.8, 2100)
        p = predict_spike_probability(self.model, self.profile, state)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_risk_reacts_to_worse_signals(self) -> None:
        better = DynamicState(130, 2, 20, 80, 45, 7.5, 7000)
        worse = DynamicState(180, 20, 38, 98, 25, 4.0, 1500)

        p_better = predict_spike_probability(self.model, self.profile, better)
        p_worse = predict_spike_probability(self.model, self.profile, worse)

        self.assertGreater(p_worse, p_better)


if __name__ == "__main__":
    unittest.main()
