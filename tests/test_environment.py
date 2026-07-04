from importlib.util import find_spec
from pathlib import Path


def test_project_structure_exists():
    assert Path('src').exists()
    assert Path('tests').exists()
    assert Path('config').exists()
    assert Path('data/raw').exists()
    assert Path('data/processed').exists()
    assert Path('outputs/tables').exists()
    assert Path('outputs/figures').exists()
    assert Path('outputs/reports').exists()


def test_core_dependencies_are_installed():
    assert find_spec('pandas') is not None
    assert find_spec('sklearn') is not None
    assert find_spec('mlflow') is not None
    assert find_spec('pandera') is not None
    assert find_spec('matplotlib') is not None
    assert find_spec('yaml') is not None
