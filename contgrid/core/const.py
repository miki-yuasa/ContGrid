from enum import Enum

from .typing import RGB

# Grid length is 1[m]

DRAG: float = 1
COLLISION_FORCE: float = 1e3
CONTACT_MARGIN: float = 1e-2
ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Color(RGB, Enum):
    RED = (213, 94, 0)
    ORANGE = (230, 159, 0)
    YELLOW = (240, 228, 66)
    GREEN = (0, 158, 115)
    BLUE = (0, 114, 178)
    SKY_BLUE = (86, 180, 233)
    PURPLE = (204, 121, 167)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREY = (127, 127, 127)
    LIGHT_GREY = (199, 199, 199)
    DARK_GREY = (77, 77, 77)
