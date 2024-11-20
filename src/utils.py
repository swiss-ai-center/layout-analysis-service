from json import JSONEncoder
import json
from paddleocr.ppstructure.utility import parse_args


def custom_parse_args(**kwargs):
    # Temporarily override `sys.argv`
    import sys  # noqa: E402
    original_argv = sys.argv
    sys.argv = ["main.py"] + [f"--{k}={v}" for k, v in kwargs.items()]

    args = parse_args()

    # Restore original argv
    sys.argv = original_argv
    return args


class CustomEncoder(JSONEncoder):
    def default(self, o):
        return json.dumps(
            o,
            default=lambda x: x.__dict__,
            sort_keys=True,
            indent=4)
