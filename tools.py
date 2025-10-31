from dataclasses import dataclass
from random import randint

@dataclass
class DispatchResult:
    resources: list
    eta_min: int

def dispatch_resources(kind: str, location: str, injuries: bool) -> DispatchResult:
    if kind == "fire":
        resources = ["firetruck", "firefighter"]
    elif kind == "medical":
        resources = ["ambulance", "doctors"]
    elif kind == "both":
        resources = ["firetruck", "ambulance", "doctors", "firefighter"]
    else:
        resources = ["emergency unit"]

    base = ["dispatcher", "communication center"]
    return DispatchResult(resources=base + resources, eta_min=randint(3, 8))
