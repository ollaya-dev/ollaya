"""GGUF packaging regressions; no network, weights or model runtimes needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollaya_convert import package


class ProjectorPackaging(unittest.TestCase):
    def test_opt_in_projector_and_source_pin(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = {'gguf': {'repo': 'author/model', 'revision': 'commit', 'path': 'model.gguf',
                            'sha256': 'abc', 'quantization': 'Q8_0'},
                   'layout': 'winnow-v1', 'llama': {'n_ctx': 8192}}
            Path(root, 'decision.json').write_text(json.dumps(cfg))
            Path(root, 'calibration.json').write_text('{"temperature": [1, 1, 1]}')
            spec = {'model': 'winnow', 'family': 'winnow', 'license': 'Apache-2.0', 'license_text': 'license'}
            variant = {'repo': 'author/model', 'commit': 'commit', 'gguf': 'model.gguf',
                       'export_dir': root, 'parameter_size': 'E4B', 'languages': ['multilingual'],
                       'description': 'text'}

            def upstream(media, repo, commit, path):
                return {'mediaType': media, 'digest': 'sha256:abc', 'size': 123,
                        'urls': [f'https://huggingface.co/{repo}/resolve/{commit}/{path}']}

            with patch.object(package, 'REGISTRY', root), \
                 patch.object(package, 'upstream', side_effect=upstream) as source, \
                 patch.object(package, 'hf_commit_date', return_value='2026-01-01'):
                blobs = package.Blobs('https://ollaya.dev')
                text_config, text_layers = package.package_gguf(spec, 'e4b', variant, blobs)
                config, layers = package.package_gguf(spec, 'e4b-vision',
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
                    package.package_gguf(spec, 'e4b-vision', {**variant, 'mmproj': 'mmproj.gguf'}, blobs)


if __name__ == '__main__':
    unittest.main()
