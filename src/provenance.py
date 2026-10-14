"""Record exact source, configuration and software used in a completed run."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_run_manifest(config: dict[str, Any], completed_steps: list[str]) -> Path:
    source = Path(config['rutas']['datos_crudos'])
    tracked_code = sorted(Path('src').glob('*.py')) + sorted(Path('config').glob('*.yaml'))
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'openpyxl', 'matplotlib', 'pandera', 'mlflow']}
    result = {
        'source_filename': source.name,
        'source_sha256': file_sha256(source),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(), 'package_versions': versions,
        'code_and_config_sha256': {str(p): file_sha256(p) for p in tracked_code},
        'completed_steps': completed_steps,
        'comparison_scope': 'descriptive, conditional on observed years; no population significance tests',
    }
    destination = Path(config['rutas']['carpeta_reportes']) / 'run_manifest.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination
