"""End-to-end image API/lifecycle check against an isolated Ollaya daemon.

python -m ollaya_convert.families.winnow.vision_http URL VISION_MODEL TEXT_MODEL
Both models must be installed. Use an isolated daemon/store; this unloads these models.
"""
import argparse
import json
import urllib.error
import urllib.request

from .vision_parity import png, post


def check(url, model, text_model):
    q = {'color': {'type': 'choice', 'instructions': 'What is the dominant color in the image?',
                   'criteria': {'red': 'Red', 'green': 'Green', 'blue': 'Blue'}}}
    base = {'model': model, 'state': 'Inspect the image.', 'questions': q}
    image = png(128, 128, (255, 0, 0))
    good = post(url, '/api/decide', {**base, 'images': [image]})
    assert good['answers']['color']['choice'] == 'red', good
    data_url = post(url, '/api/decide', {**base, 'images': ['data:image/png;base64,' + image]})
    assert data_url['answers'] == good['answers'], 'data URL changed result'
    # Compare an image-free request with the matching text-only tag below.
    without = post(url, '/api/decide', base)
    for name, images in [('base64', ['not-base64!']), ('PNG', ['AQID']),
                         ('count', [image] * 17)]:
        try:
            post(url, '/api/decide', {**base, 'images': images})
            raise AssertionError(f'{name}: invalid image request succeeded')
        except urllib.error.HTTPError as e:
            assert e.code == 422, (name, e.code, e.read())
    try:
        post(url, '/api/decide', {**base, 'images': [image], 'questions': {'long': {**q['color'], 'instructions': 'describe ' * 10000}}})
        raise AssertionError('context overflow succeeded')
    except urllib.error.HTTPError as e:
        reason = e.read().decode()
        assert e.code == 422 and 'context' in reason, (e.code, reason)
    again = post(url, '/api/decide', {**base, 'images': [image, image]})
    assert again['answers']['color']['choice'] == 'red', again
    limit = post(url, '/api/decide', {**base, 'images': [image] * 16})
    assert limit['answers']['color']['choice'] == 'red', limit
    post(url, '/api/decide', {'model': model, 'keep_alive': 0})
    text = post(url, '/api/decide', {**base, 'model': text_model})
    assert text['answers'] == without['answers'], 'text-only model answers changed'
    try:
        post(url, '/api/decide', {**base, 'model': text_model, 'images': [image]})
        raise AssertionError('text-only tag accepted images')
    except urllib.error.HTTPError as e:
        assert e.code == 422, (e.code, e.read())
    post(url, '/api/decide', {'model': text_model, 'keep_alive': 0})
    with urllib.request.urlopen(url.rstrip('/') + '/api/ps') as r:
        assert not json.load(r)['models'], 'models remain loaded'
    print('PASS: native API, data URL, multiple images, 4 rejection cases, text regression, unload')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('url'); ap.add_argument('model'); ap.add_argument('text_model')
    args = ap.parse_args()
    check(args.url, args.model, args.text_model)


if __name__ == '__main__':
    main()
