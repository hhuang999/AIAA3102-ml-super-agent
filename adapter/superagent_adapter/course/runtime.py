"""Read-only launch feasibility and bounded, non-secret runtime provenance."""
from __future__ import annotations

import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import site
import sysconfig
import shutil
import sys

from .common import digest
from .prepare import verify_inputs

_ROOT = Path(__file__).resolve().parents[2]


def _tree_hash(root, files):
    """SHA-256 of sorted relative UTF-8 paths and individual file SHA-256 values."""
    h = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        h.update(path.relative_to(root).as_posix().encode('utf-8'))
        h.update(b'\0')
        h.update(digest(path).encode('ascii'))
        h.update(b'\n')
    return h.hexdigest()


def _core_search_paths(run_dir):
    # Reconstruct child search paths rather than trusting mutable parent sys.path.
    # A relative PYTHONPATH entry is interpreted relative to the child's cwd.
    paths = [Path(run_dir).resolve()]
    for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep):
        if entry:
            paths.append((Path(run_dir) / entry).resolve())
    locations = sysconfig.get_paths()
    for name in ('stdlib', 'platstdlib'):
        if locations.get(name):
            paths.append(Path(locations[name]).resolve())
    # Python also searches the versioned stdlib zip even when absent.
    zip_name = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    paths.append(Path(sys.base_prefix) / zip_name)
    uncertain = False
    if not sys.flags.no_site:
        # site.main first adds venv site-packages, then the enabled user site,
        # then global sites. getsitepackages() already reflects venv policy.
        sites = []
        try:
            if sys.prefix != sys.base_prefix:
                sites.extend(site.getsitepackages([sys.prefix]))
            if site.ENABLE_USER_SITE:
                sites.append(site.getusersitepackages())
            sites.extend(site.getsitepackages())
        except (AttributeError, OSError, TypeError):
            uncertain = True
        for location in dict.fromkeys(x for x in sites if x):
            base = Path(location).resolve()
            if not base.is_dir():
                continue
            paths.append(base)
            for pth in sorted(base.glob('*.pth')):
                try:
                    for line in pth.read_text(encoding='utf-8').splitlines():
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if line.startswith(('import ', 'import\t')):
                            uncertain = True
                        else:
                            extra = (base / line).resolve()
                            if extra.exists():
                                paths.append(extra)
                except (OSError, UnicodeError):
                    uncertain = True
    if any(p.is_file() for p in paths):
        uncertain = True  # zip or other path hooks are not inspected
    return list(dict.fromkeys(paths)), uncertain


def _discover_core(run_dir):
    paths, uncertain = _core_search_paths(run_dir)
    for base in paths:
        package = base / 'superagent'
        if package.is_dir():
            init = package / '__init__.py'
            run = package / 'run.py'
            if init.is_file() and run.is_file():
                return {'source': 'real_core', 'status': 'found', 'origin': str(package),
                        'run_origin': str(run), 'message': 'Static source entry found; imports and dependencies are not validated.'}
            # A namespace or non-source loader requires actual import resolution.
            if not init.is_file() or not run.is_file():
                uncertain = True
                if init.is_file():
                    break
        if (base / 'superagent.py').exists():
            uncertain = True
            break
    status = 'uncertain' if uncertain else 'missing'
    return {'source': 'real_core', 'status': status, 'origin': None, 'run_origin': None,
            'message': 'Core discovery is uncertain due to unsupported import paths or loaders.' if uncertain
            else 'superagent.run was not found in the child interpreter search paths.'}


def _fixture_info(command_prefix):
    info = {'source': 'fixture', 'status': 'invalid', 'origin': None, 'run_origin': None}
    if not isinstance(command_prefix, (list, tuple)) or not command_prefix or not all(isinstance(x, str) and x for x in command_prefix):
        info['message'] = 'An explicit fixture command prefix is required.'
        return info
    exe = command_prefix[0]
    executable = shutil.which(exe) or (str(Path(exe).resolve()) if Path(exe).is_file() else None)
    if not executable or (os.name != 'nt' and not os.access(executable, os.X_OK)):
        info['message'] = 'Fixture executable is missing or not executable.'
        return info
    if len(command_prefix) > 1:
        script = Path(command_prefix[1])
        if command_prefix[1].startswith('-') or not script.is_absolute() or not script.is_file():
            info['message'] = 'Fixture script must be an existing absolute file path.'
            return info
    info.update(status='found', run_origin=str(Path(command_prefix[1]).resolve()) if len(command_prefix) > 1 else None, message='Explicit fixture executable and script checked; no code executed.')
    return info


def preflight(task_dir, private_dir, run_dir, *, config=None, execution_kind='real_core', command_prefix=None):
    checks, warnings = [], []
    def add(name, ok, message):
        checks.append({'name': name, 'ok': bool(ok), 'message': message})
    try:
        verify_inputs(task_dir, private_dir)
        add('inputs', True, 'Input manifests and hashes verified.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        add('inputs', False, f'Input verification failed: {type(exc).__name__}: {exc}')
    try:
        task, private, run = (Path(p).resolve() for p in (task_dir, private_dir, run_dir))
        separate = not run.exists() and not any(run.is_relative_to(p) or p.is_relative_to(run) for p in (task, private))
        add('run_path', separate, 'Run path must be new and disjoint from task and private directories.')
        ancestor = run
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        writable = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
        add('run_ancestor', writable, 'An existing writable directory ancestor is required; no directory was created.')
        if writable:
            warnings.append('Writable access is a static check and does not guarantee a later write will succeed.')
    except (OSError, ValueError, TypeError) as exc:
        run = Path.cwd()
        add('run_path', False, f'Invalid run path: {type(exc).__name__}')
        add('run_ancestor', False, 'Unable to check run ancestor.')
    config_hash = None
    if config is not None:
        try:
            config_hash = digest(config)
            add('config', True, 'Configuration is readable and hashed; its contents and schema were not checked.')
        except (OSError, ValueError, TypeError) as exc:
            add('config', False, f'Configuration cannot be read: {type(exc).__name__}')
    else:
        add('config', True, 'No configuration supplied.')
    python_ok = sys.version_info >= (3, 11) and Path(sys.executable).is_file() and os.access(sys.executable, os.X_OK)
    add('python', python_ok, 'Current Python must be version 3.11 or newer with an executable interpreter.')
    if execution_kind == 'real_core' and command_prefix is None:
        core = _discover_core(run)
    elif execution_kind == 'fixture':
        core = _fixture_info(command_prefix)
    else:
        core = {'source': execution_kind, 'status': 'invalid', 'origin': None, 'run_origin': None,
                'message': 'Custom command prefixes require explicit fixture execution.'}
    add('core', core['status'] == 'found', core['message'])
    warnings.append('Static discovery does not validate imports, dependencies, model access, or execution.')
    return {'ready': all(c['ok'] for c in checks), 'checks': checks, 'warnings': warnings, 'core_info': core, 'config_sha256': config_hash}


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_snapshot(*, config=None, execution_kind='real_core', command_prefix=None, core_info=None):
    core = dict(core_info) if core_info is not None else (
        _fixture_info(command_prefix) if execution_kind == 'fixture' else _discover_core(Path.cwd()))
    core.update(run_sha256=None, source_sha256=None, version=None,
                source_hash_definition='SHA-256 of sorted relative paths and file SHA-256 values: superagent/**/*.py, superagent/**/*.yaml, superagent/**/*.yml')
    if core.get('source') == 'fixture' and core.get('run_origin'):
        try:
            core['run_sha256'] = digest(core['run_origin'])
        except OSError:
            pass
    if core.get('source') == 'real_core' and core.get('status') == 'found':
        try:
            package = Path(core['origin'])
            core['run_sha256'] = digest(core['run_origin'])
            core['source_sha256'] = _tree_hash(package, [p for p in package.rglob('*') if p.suffix in ('.py', '.yaml', '.yml') and p.is_file() and not p.is_symlink()])
            core['version'] = _version('superagent')
        except (OSError, ValueError, TypeError):
            core['source_sha256'] = None
    files = [p for p in (_ROOT / 'superagent_adapter').rglob('*.py') if p.is_file() and not p.is_symlink()]
    files += [p for p in (_ROOT / 'pyproject.toml',) if p.is_file() and not p.is_symlink()]
    try:
        source_hash = _tree_hash(_ROOT, files)
    except OSError:
        source_hash = None
    try:
        # The source package declaration is authoritative for this checkout.
        import tomllib
        adapter_version = tomllib.loads((_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    except (ImportError, OSError, KeyError, ValueError):
        adapter_version = _version('superagent-adapter')
        if adapter_version is None:
            from superagent_adapter import __version__ as adapter_version
    try:
        config_hash = digest(config) if config is not None else None
    except (OSError, ValueError, TypeError):
        config_hash = None
    return {'adapter': {'version': adapter_version, 'source_sha256': source_hash,
                        'source_hash_definition': 'SHA-256 of sorted relative paths and file SHA-256 values: superagent_adapter/**/*.py and pyproject.toml when present'},
            'core': core, 'python': {'executable': sys.executable, 'version': platform.python_version(), 'platform': platform.platform()},
            'distributions': {'numpy': _version('numpy'), 'superagent': _version('superagent')},
            'config_sha256': config_hash, 'model': None,
            'limitations': ['Static source discovery does not validate imports or dependencies.', 'This snapshot is not a complete environment lock.']}
