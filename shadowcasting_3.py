"""Implement shadowcasting

This file contains an implementation of shadowcasting for a full field of
view for an observer located at any point in the grid.
"""
import abc
import functools
import itertools
import typing as ty

MAPSIZE = 24
BLACKCHAR = " "
VISCHAR = "."
WALLCHAR = "#"

Point = ty.Tuple[int, int]
PointFn = ty.Callable[[Point], Point]
Map = ty.Set[Point]


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
