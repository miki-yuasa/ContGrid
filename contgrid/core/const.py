from enum import Enum

from .typing import RGB

DRAG: float = 0.25
COLLISION_FORCE: float = 1e2
CONTACT_MARGIN: float = 1e-3
ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Color(Enum, RGB):
    RED = (228, 3, 3)
    ORANGE = (255, 140, 0)
    YELLOW = (255, 237, 0)
    GREEN = (0, 128, 38)
    BLUE = (0, 77, 255)
    PURPLE = (117, 7, 135)
    BROWN = (120, 79, 23)
    GREY = (100, 100, 100)
    LIGHT_RED = (234, 153, 153)
    LIGHT_BLUE = (90, 170, 223)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    MEDIUM_RED = (231, 80, 80)
