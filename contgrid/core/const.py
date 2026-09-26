from enum import Enum

# Grid length is 1[m].

DRAG: float = 1
COLLISION_FORCE: float = 1
CONTACT_MARGIN: float = 1e-2
ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Color(str, Enum):
    """Color definitions aligned with Seaborn's colorblind palette and utility tones."""

    # Seaborn 'colorblind' 10-color categorical palette
    BLUE = "#0173B2"
    ORANGE = "#DE8F05"
    GREEN = "#029E73"
    RED = "#D55E00"
    PURPLE = "#CC78BC"
    BROWN = "#CA9161"
    PINK = "#FBAFE4"
    GREY = "#949494"
    YELLOW = "#ECE133"
    SKY_BLUE = "#56B4E9"

    # Neutral and utility colors
    WHITE = "#FFFFFF"
    BLACK = "#000000"
    LIGHT_GREY = "#C7C7C7"
    DARK_GREY = "#4D4D4D"

