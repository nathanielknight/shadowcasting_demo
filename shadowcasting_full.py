"""Field Of View Experiments

Implementation of Field-of-View algorithm with player and mob specific
refinements.
"""
import abc
import functools
import itertools
import typing as ty
import unittest

MAPSIZE = 24
BLACKCHAR = " "
VISCHAR = "."
WALLCHAR = "#"

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


def loadobstaclemap(filename: str) -> ty.Set[Point]:
    with open(filename) as inf:
        mapsource = inf.read()
    obstacles: ty.Set[Point] = set()
    for y, mapline in enumerate(mapsource.splitlines()):
        for x, mapchar in enumerate(mapline):
            if mapchar == "#":
                obstacles.add((x, y))
    return obstacles


CORNER_CHEAT = -0.05
def endslope(u: int, v: int) -> float:
    "Top of FoV for the next slope when a scan becomes blocked"
    return (v + 0.5 + CORNER_CHEAT) / (u - 0.5)


def startslope(u: int, v: int) -> float:
    "Top of FoV when a scan resumes after being blocked."
    return (v + 0.5 + CORNER_CHEAT) / (u + 0.5)


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


class BaseTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, p: Point) -> Point:
        raise NotImplementedError

    @abc.abstractmethod
    def reverse(self, p: Point) -> Point:
        raise NotImplementedError


class OctantTransform(BaseTransform):
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


class OffsetTransform(BaseTransform):
    def __init__(self, offset: Point) -> None:
        self._offset = offset

    def __call__(self, p: Point) -> Point:
        dx, dy = self._offset
        x, y = p
        return (x + dx, y + dy)

    def reverse(self, p: Point) -> Point:
        dx, dy = self._offset
        x, y = p
        return (x - dx, y - dy)


class TestOffsetTransform(unittest.TestCase):
    def test_from_origin(self):
        origin = (0, 0)
        transform = OffsetTransform((3, 3))
        self.assertEqual(transform(origin), (3, 3))

    def test_reversing(self):
        coord_range = range(-6, 6)
        coords: ty.List[Point] = list(
            itertools.product(coord_range, coord_range)
        )
        for p in coords:
            for dp in coords:
                trn = OffsetTransform(dp)
                self.assertEqual(p, trn.reverse(trn(p)))
                self.assertEqual(p, trn(trn.reverse(p)))


class CompoundTransform(BaseTransform):
    def __init__(self, transforms: ty.Iterable[BaseTransform]) -> None:
        self._transforms = list(transforms)
        self._reversed_transforms = list(reversed(self._transforms))

    def __call__(self, p: Point) -> Point:
        return functools.reduce(lambda p, t: t(p), self._transforms, p)

    def reverse(self, p: Point) -> Point:
        return functools.reduce(
            lambda p, t: t.reverse(p), self._reversed_transforms, p
        )


class TestCompoundTransform(unittest.TestCase):
    def test_basic(self):
        t1 = OffsetTransform((1, 1))
        t2 = OffsetTransform((2, -2))
        trn = CompoundTransform([t1, t2])
        origin = (0, 0)
        self.assertEqual(trn(origin), (3, -1))
        self.assertEqual(trn.reverse(origin), (-3, 1))
        self.assertEqual(origin, trn(trn.reverse(origin)))

    def test_reversal(self):
        trn = CompoundTransform(
            [
                OffsetTransform((1, 3)),
                OctantTransform(2),
                OffsetTransform((-1, 3)),
            ]
        )
        coord_range = range(-6, 6)
        coords = list(itertools.product(coord_range, coord_range))
        for p in coords:
            self.assertEqual(p, trn.reverse(trn(p)))
            self.assertEqual(p, trn(trn.reverse(p)))


def get_fov(obstacles: Map, frompoint: Point) -> Map:
    visible: Map = set()

    orig_x, orig_y = frompoint
    origintransform = OffsetTransform((-orig_x, -orig_y))

    mapcoord: BaseTransform

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
        mapcoord = CompoundTransform(
            [origintransform, OctantTransform(octant)]
        )
        scan(1, 1.0, 0.0)

    return visible



def add_points(p1: Point, p2: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    return (x1+x2, y1+y2)

if __name__ == "__main__":
    from bearlibterminal import terminal

    position = (5, 5)
    # obstacles = {(1, 1), (2, 2), (-1, 2), (-2, 1)}
    obstacles = loadobstaclemap("testmap.txt")

    directions = {
        terminal.TK_J: (0, 1),
        terminal.TK_H: (-1, 0),
        terminal.TK_K: (0, -1),
        terminal.TK_L: (1, 0),
        terminal.TK_LEFT: (-1, 0),
        terminal.TK_RIGHT: (1, 0),
        terminal.TK_UP: (0, -1),
        terminal.TK_DOWN: (0, 1),
    }

    print("Opening terminal")
    terminal.open()
    terminal.set("window: size=64x32;")
    terminal.refresh()

    EXIT_CODES = (terminal.TK_CLOSE, terminal.TK_ESCAPE, terminal.TK_Q)
    while True:
        inp = terminal.read()
        if inp in EXIT_CODES:
            break
        direction = directions.get(inp)
        if direction is None:
            continue
        new_pos = add_points(position, direction)
        if new_pos not in obstacles:
            position = new_pos
        visible = get_fov(obstacles, position)
        terminal.clear()
        terminal.print(*position, "@")
        for ob in obstacles:
            terminal.print(*ob, "#")
        for vis in visible:
            terminal.print(*vis, ".")
        terminal.refresh()

    terminal.close()