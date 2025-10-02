from pydantic import BaseModel


class Grid(BaseModel):
    layout: list[str] | list[tuple[int, int]]
    width_cells: int
    height_cells: int
    cell_size: float = 0.100

    @property
    def width(self) -> float:
        return self.width_cells * self.cell_size

    @property
    def height(self) -> float:
        return self.height_cells * self.cell_size
