"""Real-runner regression: mtmd must not log request text, even at debug level.

Start an isolated image runner with OLLAYA_LOG=debug and redirect stdout/stderr
into LOG_FILE, then run: python -m ollaya_convert.families.winnow.vision_logs URL LOG_FILE
Requires the model/projector; does not download weights or modify log files.
"""
import argparse
from pathlib import Path
from .vision_parity import png, post


def check(url, log_file):
    state_marker = 'WinnowPrivateStateLogProbe7a9c'
    question_marker = 'WinnowPrivateQuestionLogProbe5b2d'
    result = post(url, '/decide', {
        'state': state_marker,
        'questions': {'color': {'type': 'choice',
            'instructions': 'What is the dominant image color? ' + question_marker,
            'criteria': {'red': 'Red', 'green': 'Green', 'blue': 'Blue'}}},
        'images': [png(128, 128, (255, 0, 0))],
    })
    assert len(result['questions']) == 1, 'image request did not succeed'
    contents = Path(log_file).read_text()
    assert contents, 'runner log is empty: verify the correct stdout/stderr file'
    assert state_marker not in contents, 'state leaked into runner logs'
    assert question_marker not in contents, 'question leaked into runner logs'
    assert 'add_text:' not in contents, 'mtmd prompt logger is still active'
    assert 'clip_model_loader: tensor[' not in contents, 'verbose mtmd tensor dump is still active'
    print('PASS: successful image request; no state/question, prompt or tensor dumps in runner log')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('url'); ap.add_argument('log_file')
    args = ap.parse_args()
    check(args.url, args.log_file)


if __name__ == '__main__':
    main()
