"""Takes care of saving numpy arrays of data to disk.
"""

from typing import Optional
import datetime
from pred import KeystrokePrediction, MapToKeystrokeHandler
import vis
import os
import read
import common
import keyboard


class SaveHandler(common.AbstractHandler):
    """Save the latest sample to disc."""

    def __init__(self, path: str, fail_if_exists=True):
        super().__init__()
        self.path = path
        self.fail_if_exists = fail_if_exists
        assert not (
            (self.fail_if_exists) and os.path.exists(self.path)
        ), f"Failing to save data because path `{self.path}` already exists"

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        parse_line_handler: read.ParseLineHandler = common.first_or_fail(
            [h for h in past_handlers if type(h) is read.ParseLineHandler]
        )
        now = datetime.datetime.now().isoformat(sep="T")
        insert_label_handler: Optional[vis.InsertLabelHandler] = next(iter([
            h for h in
            past_handlers if
            type(h) is vis.InsertLabelHandler
        ]), None)
        label = insert_label_handler.truth if insert_label_handler else None

        line_to_save = f"{now},{label},{','.join(parse_line_handler.raw_values[2:])}\n"
        with open(self.path, "a") as f:
            f.writelines([line_to_save])


class TypeToKeyboardHandler(common.AbstractHandler):
    def __init__(self):
        super().__init__()
        self.ctrled = {
            "m": lambda: keyboard.send("\n"),
            "j": lambda: keyboard.send("\n"),
            "h": lambda: keyboard.send("\b"),
            "[": lambda: keyboard.send("escape"),
        }
        self.shifed = {
            "1": lambda: keyboard.write("!"),
            "2": lambda: keyboard.write("@"),
            "3": lambda: keyboard.write("#"),
            "4": lambda: keyboard.write("$"),
            "5": lambda: keyboard.write("%"),
            "6": lambda: keyboard.write("^"),
            "7": lambda: keyboard.write("&"),
            "8": lambda: keyboard.write("*"),
            "9": lambda: keyboard.write("("),
            "0": lambda: keyboard.write(")"),
            "-": lambda: keyboard.write("_"),
            "=": lambda: keyboard.write("+"),
            "[": lambda: keyboard.write("{"),
            "]": lambda: keyboard.write("}"),
            ";": lambda: keyboard.write(":"),
            "'": lambda: keyboard.write('"'),
            ",": lambda: keyboard.write("<"),
            ".": lambda: keyboard.write(">"),
            "/": lambda: keyboard.write("?"),
            "\\": lambda: keyboard.write("|"),
            "`": lambda: keyboard.write("~"),
        }
        self.meta_characters = {
            'control': '⌤',
            'shift': '⇧',
            'space': ' ',
        }

    def execute(
        self,
        past_handlers: list[common.AbstractHandler],
    ):
        map_to_keystroke_handler: MapToKeystrokeHandler = common.first_or_fail(
            [h for h in past_handlers if type(h) is MapToKeystrokeHandler]
        )
        typed_keys = map_to_keystroke_handler.typed
        if len(typed_keys) > 1:
            prev_key = typed_keys[-1]
            if prev_key['keystroke'] == '':
                return
            if len(typed_keys) > 2:
                penultimate_key = typed_keys[-2]
                match penultimate_key['keystroke']:
                    case 'control':
                        # Delete the previous meta-key keystroke
                        keyboard.send("del")
                        # Send the appropriate control-modified key
                        try:
                            self.ctrled[prev_key['keystroke']]()
                        except Exception as e:
                            print(e)
                    case 'shift':
                        # Delete the previous meta-key keystroke
                        keyboard.send("del")
                        # Send the appropriate shift-modified key
                        try:
                            self.shifed[prev_key['keystroke']]()
                        except Exception as e:
                            print(e)

            # If the keystroke is a metacharacter, send a symbol for that
            # metacharacter. Otherwise just send the keystroke.
            keyboard.write(
                self.meta_characters.get(
                    prev_key['keystroke'], prev_key['keystroke']
                )
            )
