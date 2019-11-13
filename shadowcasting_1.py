"""Impement shadowcasting for one octant.

This file contains an implementation of shadow casting for points in a
rectilinear grid for the section where X and Y are positive and X is
less than Y (i.e. the octant immediately above the positive X axis).
"""
import typing as ty

MAPSIZE = 8
VISCHAR = " "
WALLCHAR = "#"

Point = ty.Tuple[int, int]
Map = ty.Set[Point]


def print_maps(maps: ty.Mapping[str, Map]):
    pmap: ty.Dict[Point, str] = dict()
    for x in range(MAPSIZE):
        for y in range(x, -1, -1):
            for c, m in maps.items():
                if (x, y) in m:
                    pmap[(x, y)] = c
    lines: ty.List[str] = list()
    for y in range(MAPSIZE):
        xs = list(range(y, MAPSIZE))
        drawnline = [pmap.get((x, y), WALLCHAR) for x in xs]
        printedline = VISCHAR * y + "".join(drawnline)
        lines.append(printedline)
    print("\n".join(reversed(lines)))


def endslope(x: int, y: int) -> float:
    "Top of FoV for the next slope when a scan becomes blocked"
    return (y + 0.5) / (x - 0.5)


def startslope(x: int, y: int) -> float:
    "Top of FoV when a scan resumes after being blocked."
    return (y + 0.5) / (x + 0.5)


def get_fov(obstacles: Map) -> Map:
    visible: Map = set()

    def scan(x: int, maxslope: float, minslope: float) -> None:

        if x >= MAPSIZE:
            return

        starty = int(x * maxslope)
        endy = max(0, round(x * minslope))

        blocked = (x, starty) in obstacles
        newmax = maxslope

        for y in range(starty, endy - 1, -1):
            if (x, y) in obstacles:
                if not blocked:
                    blocked = True
                    scan(x + 1, maxslope=newmax, minslope=endslope(x, y))
                else:
                    continue
            else:
                if blocked:
                    blocked = False
                    newmax = startslope(x, y)
                    visible.add((x, y))
                else:
                    visible.add((x, y))
        else:
            if not blocked:
                scan(x + 1, maxslope=newmax, minslope=minslope)

    scan(1, 1.0, 0.0)

    return visible


if __name__ == "__main__":
    obstacles = {(4, 1), (4, 2), (5, 1), (5, 2)}
    visible = get_fov(obstacles)
    print_maps({"@": {(0, 0)}, ".": visible, "#": obstacles})
