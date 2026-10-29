"""Complete four-station operations used by the paper protocol.

Each station receives one full circle; transfers count in time and carry no pings.
"""
import numpy as np
from risk_design.closed_tours import ClosedTour

class StationTour:
    """Four complete local surveys, with three actual straight transfers.

    One addressed station throughout each local survey. No pings on transfers,
    including a response slot intersecting a transfer. Global time never reset.
    """

    def __init__(self, family, radius, beacons, angle_deg, order):
        self.family = family
        self.radius = float(radius)
        self.angle = float(angle_deg)
        self.order = list(order)
        self.center = beacons[:, :2].mean(axis=0)
        self.version = 'station-complete-operation-v1'
        self.parts = []
        self.connector_length = 0.0
        self.length = 0.0
        local_family = family.removeprefix('station_')
        previous = None
        for index in order:
            core = ClosedTour(local_family, radius, beacons[index, :2], np.deg2rad(angle_deg))
            start = core.positions([0.0])[0, :2]
            if previous is not None:
                length = float(np.linalg.norm(start - previous))
                self.parts.append(dict(a=self.length, b=self.length + length, beacon=-1, start=previous, end=start, core=None))
                self.length += length
                self.connector_length += length
            length = core.length()
            self.parts.append(dict(a=self.length, b=self.length + length, beacon=index, core=core))
            self.length += length
            self.connector_length += core.connector_length
            previous = core.positions([length])[0, :2]

    def positions(self, distance):
        distance = np.clip(np.atleast_1d(distance), 0.0, self.length)
        xy = np.empty((len(distance), 2))
        for part in self.parts:
            mask = (distance >= part['a']) & (distance <= part['b'])
            if not mask.any():
                continue
            local = distance[mask] - part['a']
            if part['core'] is None:
                xy[mask] = part['start'] + local[:, None] / (part['b'] - part['a']) * (part['end'] - part['start'])
            else:
                xy[mask] = part['core'].positions(local)[:, :2]
        return xy

    def acquisition(self, start, end):
        ids = np.full(len(start), -1, dtype=int)
        for part in self.parts:
            if part['core'] is None:
                continue
            mask = (start >= part['a'] - 1e-08) & (end <= part['b'] + 1e-08)
            ix = np.flatnonzero(mask)
            keep = part['core'].acquisition_mask(start[ix] - part['a'], end[ix] - part['a'])
            ids[ix[keep]] = part['beacon']
        return ids

    def vessel(self, times, speed, lever):
        s = np.asarray(times) * speed
        xy = self.positions(s)
        delta = self.positions(np.minimum(s + 0.1, self.length)) - self.positions(np.maximum(s - 0.1, 0.0))
        yaw = np.arctan2(delta[:, 0], delta[:, 1])
        antenna = np.c_[xy, np.zeros(len(xy))]
        offset = np.c_[lever[0] * np.cos(yaw) + lever[1] * np.sin(yaw), -lever[0] * np.sin(yaw) + lever[1] * np.cos(yaw), np.full(len(xy), lever[2])]
        return (antenna, antenna + offset)

def station_data(tour, c, tracer):
    horizontal = tour.radius * 1.001 + np.linalg.norm(c['lever'][:2])
    slot = max(c['interval'], 2 * (horizontal + float(np.max(-c['beacons'][:, 2]))) / float(np.min(tracer.V)) + 1.0)
    times = np.arange(0.0, tour.length / c['speed'] - slot + 1e-09, slot)
    ids = tour.acquisition(times * c['speed'], (times + slot) * c['speed'])
    antenna, ship = tour.vessel(times, c['speed'], c['lever'])
    acquire = ids >= 0
    range_ok = np.ones(len(ids), dtype=bool)
    range_ok[acquire] = np.linalg.norm(ship[acquire] - c['beacons'][ids[acquire]], axis=1) <= c['max_range']
    keep = acquire & range_ok
    travel = np.full(len(times), np.nan)
    mapping = np.zeros((len(times), 3))
    ray_fail = 0
    for b, beacon in enumerate(c['beacons']):
        ix = np.flatnonzero((ids == b) & keep)
        if not len(ix):
            continue
        try:
            travel[ix] = tracer.varying_offset(ship[ix], beacon, 0.0)[0]
            mapping[ix] = tracer.mapping(ship[ix], beacon, c['center'])
        except ValueError:
            for k in ix:
                try:
                    travel[k] = tracer.varying_offset(ship[k:k + 1], beacon, 0.0)[0][0]
                    mapping[k] = tracer.mapping(ship[k:k + 1], beacon, c['center'])[0]
                except ValueError:
                    keep[k] = False
                    ray_fail += 1
    counts = np.bincount(ids[keep], minlength=len(c['beacons']))
    if np.any(counts < 8):
        raise ValueError(f'Insufficient station coverage: {counts.tolist()}')
    ix = np.flatnonzero(keep)
    return dict(times=times[ix], ids=ids[ix], ship=ship[ix], true_ship=ship[ix], antenna=antenna[ix], delays=travel[ix], travel=travel[ix], field=np.zeros(len(ix)), h=mapping[ix], slot_s=slot, per_beacon_counts=counts.tolist(), attempted=len(times), rejected_range=int((acquire & ~range_ok).sum()), rejected_ray=ray_fail, rejected_dropout=0, skipped_transfer=int((~acquire).sum()), slot_indices=ix)
