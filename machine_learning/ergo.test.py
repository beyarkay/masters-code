import unittest
import ergo as E


class TestErgo(unittest.TestCase):
    def test_serial_port(self):
        self.assertRaisesRegex(SystemExit, "1", E.get_serial_port)

    def test_gesture_info(self):
        val = E.gesture_info()
        keys = [f"gesture{i:0>4}" for i in (list(range(0, 50)) + [255])]
        self.assertEqual(list(val.keys()), keys)

    def test_format_prediction(self):
        val = E.format_prediction("gesture0042", 0.9, g2k={"gesture0042": "g"})
        self.assertEqual(val, "\x1b[30;48;2;247;131;16mg90%\x1b[0m")

        val = E.format_prediction("gesture0042", 0.9, g2k={"gesture0042": "\t"})
        self.assertEqual(val, "\x1b[30;48;2;247;131;16m\\t90%\x1b[0m")

        val = E.format_prediction("gesture0042", 0.9, g2k={"gesture0042": "\r"})
        self.assertEqual(val, "\x1b[30;48;2;247;131;16m\r90%\x1b[0m")

        val = E.format_prediction("gesture0042", 0.9, g2k={"gesture0042": "\n"})
        self.assertEqual(
            val,
            "\x1b[30;48;2;247;131;16m\\n90%\x1b[0m",
        )

        val = E.format_prediction("gesture0042", 0.9)
        self.assertEqual(val, "\x1b[30;48;2;247;131;16mg42 90%\x1b[0m")

        val = E.format_prediction("gesture0042", 0.1)
        self.assertEqual(val, "\x1b[37;48;2;15;9;45mg42 10%\x1b[0m")

        val = E.format_prediction("gesture0255", 0.9)
        self.assertEqual(val, "\x1b[30;48;2;247;131;16mg255 90%\x1b[0m")

    def test_get_colored_string(self):
        val = E.get_colored_string(20, 100)
        self.assertEqual(val, "\x1b[30;48;2;70;130;248m 20\x1b[0m")

        val = E.get_colored_string(2, 10)
        self.assertEqual(val, "\x1b[30;48;2;54;168;249m  2\x1b[0m")

    def test_color_bg(self):
        val = E.color_bg("10")
        self.assertEqual(val, "\x1b[48;2;86;39;114m10\x1b[0m")

        val = E.color_bg("abdc10")
        self.assertEqual(val, "\x1b[48;2;86;39;114mabdc10\x1b[0m")

        val = E.color_bg("gesture0001")
        self.assertEqual(val, "\x1b[48;2;66;123;94mgesture0001\x1b[0m")

        self.assertRaises(ValueError, lambda: E.color_bg("abcd"))

    def test_get_color(self):
        self.assertEqual(E.get_color("asdf123"), "rgb(63, 52, 114)")
        self.assertEqual(E.get_color("42"), "rgb(48, 153, 74)")
        self.assertEqual(E.get_color("381"), "rgb(54, 86, 121)")
        self.assertRaises(ValueError, lambda: E.get_color("blueandyellow"))

    def test_get_gesture_counts(self):
        should_be = {
            "g255": 1160,
            "g001": 13,
            "g002": 12,
            "g004": 12,
            "g000": 10,
            "g003": 10,
        }
        self.assertEqual(E.get_gesture_counts(root="mocks/"), should_be)

    def test_main(self):
        self.assertRaisesRegex(
            SystemExit,
            "1",
            lambda: E.main({"predict": False, "save": False, "as-keyboard": True}),
        )
        self.assertRaisesRegex(
            SystemExit,
            "1",
            lambda: E.main({"predict": False, "save": True, "as-keyboard": False}),
        )
        self.assertRaisesRegex(
            SystemExit,
            "1",
            lambda: E.main({"predict": True, "save": False, "as-keyboard": False}),
        )

        self.assertRaisesRegex(
            SystemExit,
            "1",
            lambda: E.main({"predict": False, "save": False, "as-keyboard": False}),
        )


if __name__ == "__main__":
    unittest.main()
