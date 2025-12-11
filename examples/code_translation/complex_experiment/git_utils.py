"""Git branch isolation for search."""
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from encompass import branchpoint

@dataclass
class GitState:
    path: Path; branch: str = "main"; commits: list = field(default_factory=list)

_s = {}

def branchpoint_git_commit(repo, name, **kw):
    if str(repo) not in _s: _s[str(repo)] = GitState(path=repo)
    _s[str(repo)].commits.append(uuid.uuid4().hex[:8])
    result = yield branchpoint(name=name, **kw)
    _s[str(repo)].branch = f"bp-{name}-{uuid.uuid4().hex[:6]}"
    return result

def reset(): global _s; _s = {}
