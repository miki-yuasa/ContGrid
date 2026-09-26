from __future__ import annotations

from absl.testing import absltest, parameterized

from contgrid.core.const import Color


class TestColor(parameterized.TestCase):

    @parameterized.named_parameters(
        ("blue", Color.BLUE, "#0173B2"),
        ("orange", Color.ORANGE, "#DE8F05"),
        ("green", Color.GREEN, "#029E73"),
        ("red", Color.RED, "#D55E00"),
        ("purple", Color.PURPLE, "#CC78BC"),
        ("brown", Color.BROWN, "#CA9161"),
        ("pink", Color.PINK, "#FBAFE4"),
        ("grey", Color.GREY, "#949494"),
        ("yellow", Color.YELLOW, "#ECE133"),
        ("sky_blue", Color.SKY_BLUE, "#56B4E9"),
    )
    def test_seaborn_colorblind_palette_hexes(
        self, color: Color, expected_hex: str
    ) -> None:
        self.assertEqual(color.value, expected_hex)

    @parameterized.named_parameters(
        ("white", Color.WHITE, "#FFFFFF"),
        ("black", Color.BLACK, "#000000"),
        ("light_grey", Color.LIGHT_GREY, "#C7C7C7"),
        ("dark_grey", Color.DARK_GREY, "#4D4D4D"),
    )
    def test_neutral_colors(self, color: Color, expected_hex: str) -> None:
        self.assertEqual(color.value, expected_hex)

    def test_color_is_str_subclass(self) -> None:
        self.assertIsInstance(Color.BLUE, str)
        self.assertEqual(Color.BLUE, "#0173B2")


if __name__ == "__main__":
    absltest.main()
