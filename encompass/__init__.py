from core.signals import BranchPoint, ScoreSignal
# Helper functions
def branchpoint(name, **kwargs):
    return BranchPoint(name, kwargs)

def record_score(value, context=""):
    return ScoreSignal(value, context)

from core.decorators import encompass_agent
# Alias for compatibility with paper/docs
compile = encompass_agent

from encompass.std import action
