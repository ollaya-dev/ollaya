"""The audio goldens' synthetic cases and prompt; no server, network or weights needed."""
import base64
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollaya_convert.families.winnow import audio_parity as ap
from ollaya_convert.families.winnow.ref import compile_request


class AudioParityCases(unittest.TestCase):
    def test_cases_are_deterministic_16k_mono_float_wavs(self):
        a, b = ap.cases(), ap.cases()
        self.assertEqual([c['id'] for c in a], [c['id'] for c in b])
        self.assertEqual(a[0]['audio'], b[0]['audio'])
        for case in a:
            for clip in case['audio']:
                raw = base64.b64decode(clip)
                self.assertEqual((raw[:4], raw[8:12]), (b'RIFF', b'WAVE'))
                self.assertEqual(struct.unpack('<HHI', raw[20:28]), (3, 1, ap.RATE))
                seconds = struct.unpack('<I', raw[40:44])[0] / 4 / ap.RATE
                self.assertTrue(0 < seconds <= 30, seconds)

    def test_media_follow_the_runner_prompt(self):
        labels = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
        prefix, _, _ = compile_request({'state': 'x', 'questions': ap.QUESTIONS}, labels, 'nonempty')
        both = ap.media_prefix(prefix, '<m>', 1, 2)
        self.assertTrue(both.endswith('Images (in order):\n<m>\nAudio (in order):\n<m>\n<m>\nState:\n"x"\n'), both)
        self.assertEqual(ap.media_prefix(prefix, '<m>', 0, 0), prefix)


if __name__ == '__main__':
    unittest.main()
