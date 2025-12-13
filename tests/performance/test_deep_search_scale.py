"""
Performance tests for deep search scaling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from core.compiler import compile_agent as compile
from core.signals import branchpoint, record_score
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch
from storage.filesystem import FileSystemStore

@pytest.fixture
def anyio_backend():
    return 'asyncio'

# Hardcoded agents for different depths (workaround for CPS limitation)
@compile
def agent_depth_10():
    total = 0
    for i in range(10):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total

@compile
def agent_depth_30():
    total = 0
    for i in range(30):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total

@compile
def agent_depth_50():
    total = 0
    for i in range(50):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total

@compile
def agent_depth_100():
    total = 0
    for i in range(100):
        choice = branchpoint(f"step_{i}")
        total += choice
    record_score(total)
    return total

@pytest.mark.anyio
@pytest.mark.parametrize("agent_factory, depth, width", [
    (agent_depth_10, 10, 5),
    (agent_depth_30, 30, 5),
    (agent_depth_50, 50, 10),
    (agent_depth_100, 100, 5),
])
async def test_deep_search_scaling(agent_factory, depth, width, tmp_path):
    """Test search scaling at various depths."""
    
    async def sampler(node, metadata=None):
        return [1, 2, 3]  # Three options
    
    engine = ExecutionEngine()
    # Use tmp_path for test isolation
    store = FileSystemStore(str(tmp_path / f"deep_d{depth}_w{width}"))
    
    strategy = BeamSearch(
        store=store, 
        engine=engine, 
        sampler=sampler, 
        width=width,
        max_depth=1000  # Ensure strategy doesn't limit us
    )
    
    nodes = await strategy.search(agent_factory)
    
    terminal = [n for n in nodes if n.is_terminal]
    max_depth_reached = max(n.depth for n in nodes)
    
    # Assertions
    assert max_depth_reached == depth, f"Expected depth {depth}, got {max_depth_reached}"
    assert len(terminal) > 0, "No terminal nodes found"
    
    # Check that we actually explored enough nodes
    # Min nodes = depth * width (roughly, for beam search)
    assert len(nodes) >= depth * width * 0.8 
