"""Implement shadowcasting for a fixed observer.

This file contains an implementation of shadowcasting for a full field of
view for an observer located at the origin.
"""
import typing as ty
import unittest

MAPSIZE = 8
BLACKCHAR = " "

Point = ty.Tuple[int, int]
PointFn = ty.Callable[[Point], Point]
Map = ty.Set[Point]


class MinMaxCoords(ty.NamedTuple):
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    @staticmethod
    def from_map(m: Map) -> "MinMaxCoords":
        min_x = min(x for x, _ in m)
        min_y = min(y for _, y in m)
        max_x = max(x for x, _ in m)
        max_y = max(y for _, y in m)
        return MinMaxCoords(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    @staticmethod
    def from_maps(ms: ty.Iterable[Map]) -> "MinMaxCoords":
        import functools

        return functools.reduce(
            lambda a, b: a.combine(b), (MinMaxCoords.from_map(m) for m in ms)
        )

    def combine(self, other: "MinMaxCoords") -> "MinMaxCoords":
        return MinMaxCoords(
            min_x=min(self.min_x, other.min_x),
            min_y=min(self.min_y, other.min_y),
            max_x=max(self.max_x, other.max_x),
            max_y=max(self.max_y, other.max_y),
        )


def print_maps(maps: ty.Mapping[str, Map]):
    from functools import reduce
    from collections import ChainMap

    pmap: ty.Mapping[Point, str] = ChainMap(
        *[{p: c} for c, ps in maps.items() for p in ps]
    )
    minmaxes = [MinMaxCoords.from_map(m) for m in maps.values()]
    minmax = reduce(lambda a, b: a.combine(b), minmaxes)
    lines: ty.List[str] = []
    for y in range(minmax.max_y, minmax.min_y - 1, -1):
        xs = range(minmax.min_x, minmax.max_x + 1)
        linechars = (pmap.get((x, y), BLACKCHAR) for x in xs)
        printedline = "".join(linechars)
        lines.append(printedline)
    print("\n".join(lines))


def endslope(u: int, v: int) -> float:
    "Top of FoV for the next slope when a scan becomes blocked"
    return (v + 0.5) / (u - 0.5)


def startslope(u: int, v: int) -> float:
    "Top of FoV when a scan resumes after being blocked."
    return (v + 0.5) / (u + 0.5)


def reflect_x(p: Point) -> Point:
    u, v = p
    return -u, v


def reflect_y(p: Point) -> Point:
    u, v = p
    return u, -v


def flip_xy(p: Point) -> Point:
    u, v = p
    return v, u


TRANSFORMS: ty.Dict[int, ty.List[PointFn]] = {
    1: [],
    2: [flip_xy],
    3: [flip_xy, reflect_y],
    4: [reflect_x],
    5: [reflect_y, reflect_x],
    6: [flip_xy, reflect_x, reflect_y],
    7: [flip_xy, reflect_x],
    8: [reflect_y],
}


class OctantTransform:
    def __init__(self, octant: int):
        assert octant in range(1, 9), "Invalid octant: {}".format(octant)
        self.octant = octant
        self.transforms: ty.List[PointFn] = TRANSFORMS[octant]

    @classmethod
    def apply_transforms(cls, p: Point, fns: ty.Iterable[PointFn]) -> Point:
        from functools import reduce

        for f in fns:
            p = f(p)
        return p

    def reverse(self, p: Point) -> Point:
        return self.apply_transforms(p, reversed(self.transforms))

    def __call__(self, p: Point) -> Point:
        return self.apply_transforms(p, self.transforms)


class TestOctantTransform(unittest.TestCase):
    def assert_in_first_octant(self, p: Point):
        x, y = p
        self.assertGreaterEqual(x, 0, "X should be positive")
        self.assertGreaterEqual(y, 0, "Y should be positive")
        self.assertGreaterEqual(x, y, "X should be >= Y")

    def test_basics(self):
        cases = {
            1: (2, 1),
            2: (1, 2),
            3: (-1, 2),
            4: (-2, 1),
            5: (-2, -1),
            6: (-1, -2),
            7: (1, -2),
            8: (2, -1),
        }
        for octant, pt in cases.items():
            trn = OctantTransform(octant)
            try:
                self.assert_in_first_octant(trn(pt))
            except AssertionError:
                print("octant={}".format(octant))
                raise

    def test_reversing(self):
        for octant in range(1, 9):
            trn = OctantTransform(octant)
            for x in range(-10, 10):
                for y in range(-10, 10):
                    p = x, y
                    self.assertEqual(p, trn.reverse(trn(p)))
                    self.assertEqual(p, trn(trn.reverse(p)))


def get_fov(obstacles: Map) -> Map:
    visible: Map = set()

    mapcoord: OctantTransform

    def scan(u: int, maxslope: float, minslope: float) -> None:
        """Apply shaodwcasting a 'column' `u`.

        Map an octant from the map into the coordinates for the first
        octant and apply shadowcasting. The mapping is applied so that
        the shadowcasting algorithm can be kept simple.
        """

        if u >= MAPSIZE:
            return

        startv = int(u * maxslope)
        endv = max(0, round(u * minslope))

        blocked = mapcoord.reverse((u, startv)) in obstacles
        newmax = maxslope

        for v in range(startv, endv - 1, -1):
            uv = (u, v)
            if mapcoord.reverse(uv) in obstacles:
                if not blocked:
                    blocked = True
                    scan(u + 1, maxslope=newmax, minslope=endslope(u, v))
                else:
                    continue
            else:
                if blocked:
                    blocked = False
                    newmax = startslope(u, v)
                    visible.add(mapcoord.reverse(uv))
                else:
                    visible.add(mapcoord.reverse(uv))
        else:
            if not blocked:
                scan(u + 1, maxslope=newmax, minslope=minslope)

    for octant in range(1, 9):
        mapcoord = OctantTransform(octant)
        scan(1, 1.0, 0.0)

    return visible


if __name__ == "__main__":
    obstacles = {(1,1), (2,2), (-1, 2), (-2, 1)}
    visible = get_fov(obstacles)
    maps = {"@": {(0, 0)}, ".": visible, "#": obstacles}
    print(MinMaxCoords.from_maps(maps.values()))
    print_maps(maps)
