import json
import math
import re
from dataclasses import asdict, dataclass

from ailias.settings import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "facefitting.json"
MODEL_SPECS = PROJECT_ROOT / "config" / "facefitting-models.json"


@dataclass(frozen=True)
class FitPolicy:
    device: str = "cpu"
    max_iterations: int = 800
    identity_candidates: tuple = (48, 96)
    expression_modes: tuple = (12, 12, 24)
    identity_regularization: float = 0.00001
    expression_regularization: float = 0.000005
    max_nfev: int = 55

    def __post_init__(self):
        if self.device != "cpu" and (not isinstance(self.device, str)
                                     or not re.fullmatch(r"cuda:[0-9]+", self.device)):
            raise ValueError("device must be explicitly cpu or cuda:N")
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 5000:
            raise ValueError("max_iterations must be an integer in 1..5000")
        for values, maximum in ((self.identity_candidates, 170), (self.expression_modes, 100)):
            if (not values or any(type(v) is not int or not 1 <= v <= maximum for v in values)):
                raise ValueError("Fit modes must be positive bounded integer sequences")
        if len(self.expression_modes) != 3 or len(set(self.identity_candidates)) != len(self.identity_candidates):
            raise ValueError("Require three expression regions and distinct identity candidates")
        for value in (self.identity_regularization, self.expression_regularization):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Fit regularization must be finite and positive")
        if type(self.max_nfev) is not int or not 1 <= self.max_nfev <= 500:
            raise ValueError("max_nfev must be an integer in 1..500")

    def record(self):
        return asdict(self)


def load_policy(path=DEFAULT_CONFIG):
    return FitPolicy(**json.loads(path.read_text(encoding="utf-8")))
