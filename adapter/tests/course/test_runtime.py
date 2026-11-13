import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from superagent_adapter.course.common import digest
from superagent_adapter.course.runtime import preflight, runtime_snapshot


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = self.root / 'task'
        self.private = self.root / 'private'
        self.task.mkdir()
        self.private.mkdir()
        (self.task / 'input.csv').write_text('id\n1\n', encoding='utf-8')
        (self.private / 'test_labels.csv').write_text('id,y\n1,0\n', encoding='utf-8')
        (self.private / 'manifest.json').write_text(json.dumps({
            'public_hashes': {'input.csv': digest(self.task / 'input.csv')},
            'private_labels_sha256': digest(self.private / 'test_labels.csv'),
        }), encoding='utf-8')
        self.run = self.root / 'run'
        self.fixture = self.root / 'fixture.py'
        self.fixture.write_text('raise RuntimeError("must not run")', encoding='utf-8')

    def check(self, **kwargs):
        return preflight(self.task, self.private, self.run, **kwargs)

    def test_fixture_is_static_and_creates_nothing(self):
        with patch('subprocess.Popen', side_effect=AssertionError('launched')):
            result = self.check(execution_kind='fixture', command_prefix=[sys.executable, str(self.fixture)])
        self.assertTrue(result['ready'], result)
        self.assertEqual(result['core_info']['source'], 'fixture')
        self.assertFalse(self.run.exists())
        json.dumps(result)

    def test_tamper_missing_config_and_existing_run(self):
        (self.task / 'input.csv').write_text('tampered', encoding='utf-8')
        result = self.check(execution_kind='fixture', command_prefix=[sys.executable, str(self.fixture)], config=self.root / 'missing')
        self.assertFalse(result['ready'])
        failed = {c['name'] for c in result['checks'] if not c['ok']}
        self.assertIn('inputs', failed)
        self.assertIn('config', failed)
        self.run.mkdir()
        self.assertFalse(self.check(execution_kind='fixture', command_prefix=[sys.executable, str(self.fixture)])['ready'])

    def test_missing_core_and_invalid_custom_command(self):
        with patch('superagent_adapter.course.runtime._core_search_paths', return_value=([], False)):
            result = self.check()
        self.assertFalse(result['ready'])
        self.assertEqual(result['core_info']['status'], 'missing')
        self.assertFalse(self.check(command_prefix=[sys.executable, str(self.fixture)])['ready'])
        self.assertFalse(self.check(execution_kind='fixture')['ready'])

    def test_cwd_only_package_not_discovered(self):
        fake = self.root / 'cwd'
        (fake / 'superagent').mkdir(parents=True)
        (fake / 'superagent' / '__init__.py').write_text('raise RuntimeError()', encoding='utf-8')
        (fake / 'superagent' / 'run.py').write_text('', encoding='utf-8')
        previous = Path.cwd()
        try:
            os.chdir(fake)
            with patch.dict(os.environ, {'PYTHONPATH': ''}), patch.object(sys, 'path', ['', str(fake)]), \
                    patch('superagent_adapter.course.runtime.sysconfig.get_paths', return_value={}), \
                    patch('superagent_adapter.course.runtime.site.getsitepackages', return_value=[]), \
                    patch('superagent_adapter.course.runtime.site.ENABLE_USER_SITE', False):
                result = self.check()
        finally:
            os.chdir(previous)
        self.assertFalse(result['ready'])
        self.assertEqual(result['core_info']['status'], 'missing')


    def test_parent_sys_path_injection_is_not_a_child_path(self):
        fake = self.root / 'injected'
        package = fake / 'superagent'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / 'run.py').write_text('', encoding='utf-8')
        with patch.dict(os.environ, {'PYTHONPATH': ''}), patch.object(sys, 'path', [str(fake)]), \
                patch('superagent_adapter.course.runtime.sysconfig.get_paths', return_value={}), \
                patch('superagent_adapter.course.runtime.site.getsitepackages', return_value=[]), \
                patch('superagent_adapter.course.runtime.site.ENABLE_USER_SITE', False):
            result = self.check()
        self.assertFalse(result['ready'])
        self.assertNotEqual(result['core_info']['status'], 'found')

    def test_user_site_shadows_global_core(self):
        user = self.root / 'user_site'
        global_site = self.root / 'global_site'
        user.mkdir()
        (global_site / 'superagent').mkdir(parents=True)
        (user / 'superagent.py').write_text('', encoding='utf-8')
        (global_site / 'superagent' / '__init__.py').write_text('', encoding='utf-8')
        (global_site / 'superagent' / 'run.py').write_text('', encoding='utf-8')
        with patch.dict(os.environ, {'PYTHONPATH': ''}), \
                patch('superagent_adapter.course.runtime.sysconfig.get_paths', return_value={'purelib': str(global_site), 'platlib': str(global_site)}), \
                patch('superagent_adapter.course.runtime.site.getsitepackages', return_value=[str(global_site)]), \
                patch('superagent_adapter.course.runtime.site.getusersitepackages', return_value=str(user)), \
                patch('superagent_adapter.course.runtime.site.ENABLE_USER_SITE', True), \
                patch('superagent_adapter.course.runtime.sys.prefix', sys.base_prefix):
            result = self.check()
        self.assertFalse(result['ready'], result)
        self.assertNotEqual(result['core_info']['status'], 'found')

    def test_venv_site_precedes_user_site(self):
        user = self.root / 'user_site'
        env = self.root / 'venv_site'
        user.mkdir()
        (env / 'superagent').mkdir(parents=True)
        (user / 'superagent.py').write_text('', encoding='utf-8')
        (env / 'superagent' / '__init__.py').write_text('', encoding='utf-8')
        (env / 'superagent' / 'run.py').write_text('', encoding='utf-8')
        with patch.dict(os.environ, {'PYTHONPATH': ''}), \
                patch('superagent_adapter.course.runtime.sysconfig.get_paths', return_value={'purelib': str(env)}), \
                patch('superagent_adapter.course.runtime.site.getsitepackages', return_value=[str(env)]), \
                patch('superagent_adapter.course.runtime.site.getusersitepackages', return_value=str(user)), \
                patch('superagent_adapter.course.runtime.site.ENABLE_USER_SITE', True), \
                patch('superagent_adapter.course.runtime.sys.prefix', str(self.root / 'venv')):
            result = self.check()
        self.assertTrue(result['ready'], result)
        self.assertEqual(result['core_info']['origin'], str(env / 'superagent'))

    def test_source_hashes_change_with_temporary_code(self):
        root = self.root / 'adapter'
        source = root / 'superagent_adapter' / 'course'
        source.mkdir(parents=True)
        code = source / 'sample.py'
        code.write_text('a = 1', encoding='utf-8')
        (root / 'pyproject.toml').write_text('[project]\nversion = "1.0"\n', encoding='utf-8')
        pkg = self.root / 'package'
        pkg.mkdir()
        run = pkg / 'run.py'
        run.write_text('a = 1', encoding='utf-8')
        resource = pkg / 'resources' / 'default_config.yaml'
        resource.parent.mkdir()
        resource.write_text('setting: one\n', encoding='utf-8')
        info = {'source': 'real_core', 'status': 'found', 'origin': str(pkg), 'run_origin': str(run)}
        with patch('superagent_adapter.course.runtime._ROOT', root):
            first = runtime_snapshot(core_info=info)
            code.write_text('a = 2', encoding='utf-8')
            run.write_text('a = 2', encoding='utf-8')
            second = runtime_snapshot(core_info=info)
            resource.write_text('setting: two\n', encoding='utf-8')
            third = runtime_snapshot(core_info=info)
        self.assertNotEqual(first['adapter']['source_sha256'], second['adapter']['source_sha256'])
        self.assertNotEqual(first['core']['source_sha256'], second['core']['source_sha256'])
        self.assertNotEqual(first['core']['run_sha256'], second['core']['run_sha256'])
        self.assertNotEqual(second['core']['source_sha256'], third['core']['source_sha256'])
        self.assertEqual(second['core']['run_sha256'], third['core']['run_sha256'])
        self.assertIn('*.yaml', third['core']['source_hash_definition'])

    def test_installed_distribution_version_fallback(self):
        root = self.root / 'site-packages'
        source = root / 'superagent_adapter'
        source.mkdir(parents=True)
        (source / '__init__.py').write_text('', encoding='utf-8')
        with patch('superagent_adapter.course.runtime._ROOT', root), \
                patch('superagent_adapter.course.runtime._version', return_value='0.3.0'):
            snap = runtime_snapshot(execution_kind='fixture', command_prefix=[sys.executable, str(self.fixture)])
        self.assertEqual(snap['adapter']['version'], '0.3.0')
        self.assertEqual(len(snap['adapter']['source_sha256']), 64)

    def test_absolute_pythonpath_discovers_core_without_import(self):
        pkg = self.root / 'core' / 'superagent'
        pkg.mkdir(parents=True)
        (pkg / '__init__.py').write_text('raise RuntimeError("executed")', encoding='utf-8')
        (pkg / 'run.py').write_text('pass', encoding='utf-8')
        with patch.dict(os.environ, {'PYTHONPATH': str(pkg.parent)}):
            result = self.check()
        self.assertTrue(result['ready'], result)
        self.assertEqual(result['core_info']['origin'], str(pkg))

    def test_python_310_is_not_ready(self):
        with patch('superagent_adapter.course.runtime.sys.version_info', (3, 10)):
            result = self.check(execution_kind='fixture', command_prefix=[sys.executable, str(self.fixture)])
        self.assertFalse(next(c for c in result['checks'] if c['name'] == 'python')['ok'])

    def test_snapshot_hashes_config_and_live_source_without_contents(self):
        config = self.root / 'config.yaml'
        config.write_text('secret: sensitive-value\nmodel: arbitrary\n', encoding='utf-8')
        snap = runtime_snapshot(config=config, execution_kind='fixture', command_prefix=[sys.executable, str(self.fixture)])
        self.assertEqual(snap['config_sha256'], digest(config))
        self.assertEqual(snap['core']['source'], 'fixture')
        self.assertEqual(snap['core']['run_sha256'], digest(self.fixture))
        self.assertEqual(snap['model'], None)
        self.assertNotIn('sensitive-value', json.dumps(snap))
        self.assertEqual(len(snap['adapter']['source_sha256']), 64)

    def test_core_discovery_does_not_execute_init(self):
        pkg = self.root / 'lib' / 'superagent'
        pkg.mkdir(parents=True)
        (pkg / '__init__.py').write_text('raise RuntimeError("executed")', encoding='utf-8')
        (pkg / 'run.py').write_text('pass', encoding='utf-8')
        with patch('superagent_adapter.course.runtime._core_search_paths', return_value=([pkg.parent], False)):
            result = self.check()
        self.assertTrue(result['ready'], result)
        self.assertEqual(result['core_info']['status'], 'found')
        snap = runtime_snapshot(core_info=result['core_info'])
        self.assertEqual(snap['core']['run_sha256'], digest(pkg / 'run.py'))


if __name__ == '__main__':
    unittest.main()
