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
        self.shifted = {
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
            "a": lambda: keyboard.write("A"),
            "b": lambda: keyboard.write("B"),
            "c": lambda: keyboard.write("C"),
            "d": lambda: keyboard.write("D"),
            "e": lambda: keyboard.write("E"),
            "f": lambda: keyboard.write("F"),
            "g": lambda: keyboard.write("G"),
            "h": lambda: keyboard.write("H"),
            "i": lambda: keyboard.write("I"),
            "j": lambda: keyboard.write("J"),
            "k": lambda: keyboard.write("K"),
            "l": lambda: keyboard.write("L"),
            "m": lambda: keyboard.write("M"),
            "n": lambda: keyboard.write("N"),
            "o": lambda: keyboard.write("O"),
            "p": lambda: keyboard.write("P"),
            "q": lambda: keyboard.write("Q"),
            "r": lambda: keyboard.write("R"),
            "s": lambda: keyboard.write("S"),
            "t": lambda: keyboard.write("T"),
            "u": lambda: keyboard.write("U"),
            "v": lambda: keyboard.write("V"),
            "w": lambda: keyboard.write("W"),
            "x": lambda: keyboard.write("X"),
            "y": lambda: keyboard.write("Y"),
            "z": lambda: keyboard.write("Z"),
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
            trimmed_typed_keys = [
                k for k in typed_keys if k['gesture_index'] != 50
            ]

            if prev_key['keystroke'] == '':
                return
            capture_keystroke = False

            if len(trimmed_typed_keys) > 2:
                penultimate_key = trimmed_typed_keys[-2]
                match penultimate_key['keystroke']:
                    case 'control':
                        # Delete the previous meta-key keystroke
                        keyboard.send("del")
                        # Send the appropriate control-modified key
                        try:
                            self.ctrled[prev_key['keystroke']]()
                            capture_keystroke = True
                        except Exception as e:
                            pass
                            # print('couldn"t control: ', e)
                    case 'shift':
                        # Delete the previous meta-key keystroke
                        keyboard.send("del")
                        # Send the appropriate shift-modified key
                        try:
                            self.shifted[prev_key['keystroke']]()
                            capture_keystroke = True
                        except Exception as e:
                            pass
                            # print('couldn"t shift: ', e)

            if not capture_keystroke:
                # If the keystroke is a metacharacter, send a symbol for that
                # metacharacter. Otherwise just send the keystroke.
                keyboard.write(
                    self.meta_characters.get(
                        prev_key['keystroke'], prev_key['keystroke']
                    )
                )
