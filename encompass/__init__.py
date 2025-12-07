from core.signals import BranchPoint, ScoreSignal, Effect
# Helper functions
def branchpoint(name, **kwargs):
    return BranchPoint(name, kwargs)

def record_score(value, context=""):
    return ScoreSignal(value, context)

from core.signals import effect

from core.decorators import encompass_agent
# Alias for compatibility with paper/docs
compile = encompass_agent

from encompass.std import action
