"""GGUF packaging regressions; no network, weights or model runtimes needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollaya_convert import catalog, package


class ProjectorPackaging(unittest.TestCase):
    def test_d1_keeps_vision_opt_in_and_uses_original_author_files(self):
        spec = catalog.CATALOG['d1']
        self.assertEqual(spec['aliases']['latest'], '3b')
        text, vision = spec['tags']['3b'], spec['tags']['3b-vision']
        self.assertNotIn('mmproj', text)
        self.assertEqual(vision['repo'], 'LiquidAI/d1-3B-GGUF')
        self.assertEqual(vision['commit'], 'bb1e436ea78eb96a3f1acb6da865f70c2fbeb563')
        self.assertEqual(vision['mmproj'], 'mmproj-d1-3B-Q8_0.gguf')
        for key in ['repo', 'commit', 'gguf', 'export_dir']:
            self.assertEqual(text[key], vision[key])

    def test_credence_variants_share_the_verified_frozen_projector(self):
        spec = catalog.CATALOG['credence']
        release = json.loads((Path(__file__).resolve().parents[1] /
                              'releases/credence-e4b.json').read_text())
        projector = release['vision_projector']
        for tag in ['e4b', 'e4b-calibrated']:
            text, vision = spec['tags'][tag], spec['tags'][tag + '-vision']
            for key in ['repo', 'commit', 'gguf', 'export_dir']:
                self.assertEqual(text[key], vision[key])
            self.assertNotIn('mmproj', text)
            # Winnow-E4B's projector, from EldanRing's repo, not the copy in Txoka's.
            self.assertEqual(vision['mmproj'], (projector['repo'], projector['revision'], projector['path']))
            self.assertEqual(projector['repo'], 'EldanRing/Winnow-E4B')

    def test_catalog_projectors_match_their_text_sources(self):
        spec = catalog.CATALOG["winnow"]
        self.assertEqual(spec["aliases"]["latest"], "12b")
        for tag, family in [("e4b", "E4B"), ("12b", "12B")]:
            with self.subTest(tag=tag):
                text = spec["tags"][tag]
                vision = spec["tags"][tag + "-vision"]
                for key in ["repo", "commit", "gguf", "export_dir", "parameter_size"]:
                    self.assertEqual(vision[key], text[key])
                self.assertNotIn("mmproj", text)
                self.assertEqual(vision["mmproj"], f"gguf/mmproj-Winnow-{family}.gguf")

    def test_opt_in_projector_and_source_pin(self):
        for tag in ["e4b", "12b"]:
            with self.subTest(tag=tag):
                self.check_projector_and_source_pin(tag)

    def check_projector_and_source_pin(self, tag):
        with tempfile.TemporaryDirectory() as root:
            cfg = {'gguf': {'repo': 'author/model', 'revision': 'commit', 'path': 'model.gguf',
                            'sha256': 'abc', 'quantization': 'Q8_0'},
                   'layout': 'winnow-v1', 'llama': {'n_ctx': 8192}}
            Path(root, 'decision.json').write_text(json.dumps(cfg))
            Path(root, 'calibration.json').write_text('{"temperature": [1, 1, 1]}')
            spec = {'model': 'winnow', 'family': 'winnow', 'license': 'Apache-2.0', 'license_text': 'license'}
            variant = {'repo': 'author/model', 'commit': 'commit', 'gguf': 'model.gguf',
                       'export_dir': root, 'parameter_size': 'E4B' if tag == 'e4b' else '12B', 'languages': ['multilingual'],
                       'description': 'text'}

            def upstream(media, repo, commit, path):
                return {'mediaType': media, 'digest': 'sha256:abc', 'size': 123,
                        'urls': [f'https://huggingface.co/{repo}/resolve/{commit}/{path}']}

            with patch.object(package, 'REGISTRY', root), \
                 patch.object(package, 'upstream', side_effect=upstream) as source, \
                 patch.object(package, 'hf_commit_date', return_value='2026-01-01'):
                blobs = package.Blobs('https://ollaya.dev')
                text_config, text_layers = package.package_gguf(spec, tag, variant, blobs)
                config, layers = package.package_gguf(spec, tag + '-vision',
                                                     {**variant, 'mmproj': 'mmproj.gguf'}, blobs)
                self.assertEqual(config, text_config)
                self.assertEqual(layers[:-1], text_layers)
                self.assertEqual(layers[-1]['mediaType'], package.MEDIA['mmproj'])
                self.assertEqual(layers[-1]['urls'],
                                 ['https://huggingface.co/author/model/resolve/commit/mmproj.gguf'])
                source.assert_any_call(package.MEDIA['mmproj'], 'author/model', 'commit', 'mmproj.gguf')
                self.assertFalse(any(p.suffix == '.gguf' for p in Path(root).rglob('*')))
                cfg['gguf']['revision'] = 'different-commit'
                Path(root, 'decision.json').write_text(json.dumps(cfg))
                with self.assertRaises(SystemExit):
                    package.package_gguf(spec, tag + '-vision', {**variant, 'mmproj': 'mmproj.gguf'}, blobs)

    def test_projector_from_another_repo(self):
        # A fine-tune that keeps its base's projector points at the author's file, not a copy.
        with tempfile.TemporaryDirectory() as root:
            cfg = {'gguf': {'repo': 'tuner/model', 'revision': 'commit', 'path': 'model.gguf',
                            'sha256': 'abc', 'quantization': 'Q8_0'},
                   'layout': 'winnow-v1', 'llama': {'n_ctx': 8192}}
            Path(root, 'decision.json').write_text(json.dumps(cfg))
            Path(root, 'calibration.json').write_text('{"temperature": [1, 1, 1]}')
            spec = {'model': 'tuned', 'family': 'winnow', 'license': 'Apache-2.0', 'license_text': 'license'}
            variant = {'repo': 'tuner/model', 'commit': 'commit', 'gguf': 'model.gguf', 'export_dir': root,
                       'parameter_size': '7.5B', 'languages': ['multilingual'], 'description': 'text',
                       'mmproj': ('author/base', 'base-commit', 'gguf/mmproj.gguf')}

            def upstream(media, repo, commit, path):
                return {'mediaType': media, 'digest': 'sha256:abc', 'size': 123,
                        'urls': [f'https://huggingface.co/{repo}/resolve/{commit}/{path}']}

            with patch.object(package, 'REGISTRY', root), \
                 patch.object(package, 'upstream', side_effect=upstream), \
                 patch.object(package, 'hf_commit_date', return_value='2026-01-01'):
                _, layers = package.package_gguf(spec, 'e4b-vision', variant, package.Blobs('https://ollaya.dev'))
                self.assertEqual(layers[0]['urls'], ['https://huggingface.co/tuner/model/resolve/commit/model.gguf'])
                self.assertEqual(layers[-1]['urls'],
                                 ['https://huggingface.co/author/base/resolve/base-commit/gguf/mmproj.gguf'])


if __name__ == '__main__':
    unittest.main()
