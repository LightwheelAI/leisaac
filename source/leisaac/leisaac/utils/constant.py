import os

try:
    from git import Repo

    repo = Repo(os.getcwd(), search_parent_directories=True)
    git_root = repo.git.rev_parse("--show-toplevel")
except Exception:
    from pathlib import Path

    git_root = Path(os.path.abspath(__file__)).parent.parent.parent.parent.parent

ASSETS_ROOT = os.path.join(git_root, 'assets')
