from pydantic import BaseModel


class Grid(BaseModel):
    layout: list[str] | list[list[str]] | list[tuple[int, int]]
    width_cells: int
    height_cells: int
    cell_size: float = 0.100

    @property
    def width(self) -> float:
        return self.width_cells * self.cell_size

    @property
    def height(self) -> float:
        return self.height_cells * self.cell_size


DEFAULT_GRID = Grid(
    layout=["#####", "#000#", "#000#", "#000#", "#####"],
    width_cells=10,
    height_cells=10,
    cell_size=0.100,
)
