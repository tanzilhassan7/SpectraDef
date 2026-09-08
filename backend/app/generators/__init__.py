from .base import AdversarialGenerator
from .fgsm import FGSMGenerator
from .one_step_targeted import TargetedOneStepGenerator
from .iterative import IterativeGenerator
from .iterative_target import IterativeTargetGenerator

GENERATORS = {
    "fgsm": FGSMGenerator,
    "one_step_targeted": TargetedOneStepGenerator,
    "iterative": IterativeGenerator,
    "iterative_target": IterativeTargetGenerator,
}
