import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable


Vector3 = Tuple[float, float, float]


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a: Vector3, s: float) -> Vector3:
    return (a[0] * s, a[1] * s, a[2] * s)


def norm(a: Vector3) -> float:
    return math.sqrt(dot(a, a))


def normalize_xy(a: Vector3) -> Vector3:
    xy = math.sqrt(a[0] * a[0] + a[1] * a[1])
    if xy == 0:
        return (1.0, 0.0, 0.0)
    return (a[0] / xy, a[1] / xy, 0.0)


def distance_point_to_segment(p: Vector3, a: Vector3, b: Vector3) -> float:
    ab = sub(b, a)
    ap = sub(p, a)
    ab_len2 = dot(ab, ab)
    if ab_len2 == 0.0:
        return norm(ap)
    t = max(0.0, min(1.0, dot(ap, ab) / ab_len2))
    proj = add(a, mul(ab, t))
    return norm(sub(p, proj))


@dataclass
class Missile:
    name: str
    position0: Vector3
    speed: float
    target_point: Vector3  # direction points towards this target

    def velocity(self) -> Vector3:
        direction = sub(self.target_point, self.position0)
        direction_norm = norm(direction)
        if direction_norm == 0:
            return (0.0, 0.0, 0.0)
        unit = (direction[0] / direction_norm, direction[1] / direction_norm, direction[2] / direction_norm)
        return mul(unit, self.speed)

    def position(self, t: float) -> Vector3:
        v = self.velocity()
        return add(self.position0, mul(v, t))


@dataclass
class UAV:
    name: str
    position0: Vector3
    speed: float  # 70..140 m/s
    heading_dir_xy: Vector3  # unit in xy plane, z=0

    def position(self, t: float) -> Vector3:
        # Constant altitude
        return (
            self.position0[0] + self.speed * self.heading_dir_xy[0] * t,
            self.position0[1] + self.speed * self.heading_dir_xy[1] * t,
            self.position0[2],
        )


@dataclass
class SmokeEvent:
    explode_time: float
    explode_center: Vector3
    radius: float = 10.0
    sink_speed: float = 3.0  # m/s downward
    life_time: float = 20.0

    def center(self, t: float) -> Optional[Vector3]:
        if t < self.explode_time or t > self.explode_time + self.life_time:
            return None
        dt = t - self.explode_time
        return (self.explode_center[0], self.explode_center[1], self.explode_center[2] - self.sink_speed * dt)


def los_blocked_by_smoke_at_time(missile_pos: Vector3, target_point: Vector3, smoke_center: Vector3, smoke_radius: float) -> bool:
    d = distance_point_to_segment(smoke_center, missile_pos, target_point)
    return d <= smoke_radius


def compute_shielding_time(missile: Missile, target_point: Vector3, smokes: List[SmokeEvent], t_start: float = 0.0, t_end: float = 80.0, dt: float = 0.02) -> float:
    # Compute total time where LoS is blocked by at least one smoke
    total = 0.0
    t = t_start
    last_block = False
    while t <= t_end:
        mpos = missile.position(t)
        blocked = False
        for s in smokes:
            c = s.center(t)
            if c is None:
                continue
            if los_blocked_by_smoke_at_time(mpos, target_point, c, s.radius):
                blocked = True
                break
        if blocked:
            total += dt
        t += dt
    return total


def shielding_profile(missile: Missile, target_point: Vector3, smokes: List[SmokeEvent], t_start: float = 0.0, t_end: float = 80.0, dt: float = 0.02) -> List[Tuple[float, int]]:
    # Returns time series of (t, blocked_count)
    result: List[Tuple[float, int]] = []
    t = t_start
    while t <= t_end:
        mpos = missile.position(t)
        count = 0
        for s in smokes:
            c = s.center(t)
            if c is None:
                continue
            if los_blocked_by_smoke_at_time(mpos, target_point, c, s.radius):
                count += 1
        result.append((t, count))
        t += dt
    return result


def compute_explosion_from_release(uav: UAV, release_time: float, fuse_time: float, g: float = 9.8) -> Tuple[float, Vector3, Vector3]:
    # Returns (explode_time, release_pos, explode_center)
    release_pos = uav.position(release_time)
    v0 = (uav.speed * uav.heading_dir_xy[0], uav.speed * uav.heading_dir_xy[1], 0.0)
    explode_time = release_time + fuse_time
    explode_center = (
        release_pos[0] + v0[0] * fuse_time,
        release_pos[1] + v0[1] * fuse_time,
        release_pos[2] - 0.5 * g * fuse_time * fuse_time,
    )
    return explode_time, release_pos, explode_center


def heading_towards_xy(from_pos: Vector3, to_xy: Tuple[float, float]) -> Vector3:
    dir_xy = (to_xy[0] - from_pos[0], to_xy[1] - from_pos[1], 0.0)
    return normalize_xy(dir_xy)


def heading_angle_deg_from_dir(dir_xy: Vector3) -> float:
    return math.degrees(math.atan2(dir_xy[1], dir_xy[0]))


def find_point_on_los(missile_pos: Vector3, target_point: Vector3, u_param: float) -> Vector3:
    # u in [0,1], 0 at missile, 1 at target
    return (
        missile_pos[0] + (target_point[0] - missile_pos[0]) * u_param,
        missile_pos[1] + (target_point[1] - missile_pos[1]) * u_param,
        missile_pos[2] + (target_point[2] - missile_pos[2]) * u_param,
    )


def plan_single_smoke_center_on_los(missile: Missile, target_point: Vector3, explode_time: float, u_on_segment: float = 0.5) -> SmokeEvent:
    t_mid = explode_time + 10.0  # mid-life of smoke
    mpos_mid = missile.position(t_mid)
    point_on_los = find_point_on_los(mpos_mid, target_point, max(0.0, min(1.0, u_on_segment)))
    return SmokeEvent(explode_time=explode_time, explode_center=point_on_los)


def uav_params_to_reach_smoke_center(uav_start: Vector3, uav_altitude: float, desired_explode_time: float, desired_center: Vector3, speed_bounds: Tuple[float, float] = (70.0, 140.0), g: float = 9.8) -> Optional[Dict[str, float]]:
    # Solve for heading, speed, release_time, fuse_time such that:
    # explode_center_xy = uav_xy0 + speed * (release_time + fuse_time) * dir_xy
    # explode_center_z = uav_altitude - 0.5*g*fuse_time^2
    # Constraints: speed in bounds, release_time >= 0, fuse_time > 0
    # Choose dir_xy towards desired_center_xy
    desired_xy = (desired_center[0], desired_center[1])
    dir_xy = heading_towards_xy(uav_start, desired_xy)
    dx = desired_xy[0] - uav_start[0]
    dy = desired_xy[1] - uav_start[1]
    dist_xy = math.hypot(dx, dy)
    # Determine fuse_time from altitude drop
    dz = uav_altitude - desired_center[2]
    if dz <= 0:
        # Desired center above or at UAV altitude is not reachable with zero initial vertical velocity
        return None
    fuse_time = math.sqrt(2.0 * dz / g)
    # L = release_time + fuse_time = dist_xy / speed
    # We can choose speed within bounds and then solve release_time = L - fuse_time >= 0
    # To satisfy release_time >= 0: L >= fuse_time => dist_xy/speed >= fuse_time => speed <= dist_xy / fuse_time
    max_speed_from_release = dist_xy / fuse_time if fuse_time > 0 else float('inf')
    speed_upper = min(speed_bounds[1], max_speed_from_release)
    speed_lower = speed_bounds[0]
    if speed_upper < speed_lower:
        # Not feasible, try allowing release_time == 0 by increasing fuse_time? Not allowed; dz fixed by desired z
        # Could slightly adjust desired z upward by 1e-6, but return None for now
        return None
    # Choose speed as mid between lower and upper
    speed = 0.5 * (speed_lower + speed_upper)
    L = dist_xy / speed
    release_time = L - fuse_time
    if release_time < 0:
        # Clamp to 0 by increasing speed slightly
        release_time = 0.0
        L = fuse_time
        speed = dist_xy / L if L > 0 else speed_bounds[1]
        if speed < speed_bounds[0] or speed > speed_bounds[1]:
            return None
    # Check explode_time consistency: explode_time = release_time + fuse_time
    explode_time = release_time + fuse_time
    # We can shift both release and explode earlier or later by the same delta by waiting before starting; here we keep explode_time as computed
    heading_deg = heading_angle_deg_from_dir(dir_xy)
    return {
        "heading_deg": heading_deg,
        "speed": speed,
        "release_time": release_time,
        "fuse_time": fuse_time,
        "explode_time": explode_time,
    }


# Problem-specific constants and helpers

TRUE_TARGET_CENTER = (0.0, 200.0, 5.0)  # approximate center of cylinder
DUMMY_TARGET = (0.0, 0.0, 0.0)

MISSILES_DEFAULT = {
    "M1": Missile(name="M1", position0=(20000.0, 0.0, 2000.0), speed=300.0, target_point=DUMMY_TARGET),
    "M2": Missile(name="M2", position0=(19000.0, 600.0, 2100.0), speed=300.0, target_point=DUMMY_TARGET),
    "M3": Missile(name="M3", position0=(18000.0, -600.0, 1900.0), speed=300.0, target_point=DUMMY_TARGET),
}

UAVS_DEFAULT = {
    "FY1": (17800.0, 0.0, 1800.0),
    "FY2": (12000.0, 1400.0, 1400.0),
    "FY3": (6000.0, -3000.0, 700.0),
    "FY4": (11000.0, 2000.0, 1800.0),
    "FY5": (13000.0, -2000.0, 1300.0),
}


def q1_compute_shield_time() -> float:
    # FY1 flies toward dummy target at 120 m/s, drop at 1.5 s, explode after 3.6 s
    fy1_start = UAVS_DEFAULT["FY1"]
    heading = heading_towards_xy(fy1_start, (0.0, 0.0))
    fy1 = UAV(name="FY1", position0=fy1_start, speed=120.0, heading_dir_xy=heading)
    explode_time, release_pos, explode_center = compute_explosion_from_release(fy1, 1.5, 3.6)
    smoke = SmokeEvent(explode_time=explode_time, explode_center=explode_center)
    m1 = MISSILES_DEFAULT["M1"]
    shield_time = compute_shielding_time(m1, TRUE_TARGET_CENTER, [smoke], t_start=0.0, t_end=80.0, dt=0.02)
    return shield_time


def grid_search_q2(direction_grid_deg: int = 72, speed_vals: List[float] = [70, 90, 110, 130, 140], release_times: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0], fuse_times: List[float] = [2.0, 2.5, 3.0, 3.5, 4.0]) -> Dict[str, float]:
    # FY1 single smoke to maximize shielding time
    fy1_start = UAVS_DEFAULT["FY1"]
    best = {"score": -1.0}
    m1 = MISSILES_DEFAULT["M1"]
    for k in range(direction_grid_deg):
        angle_deg = 360.0 * k / direction_grid_deg
        angle_rad = math.radians(angle_deg)
        dir_xy = (math.cos(angle_rad), math.sin(angle_rad), 0.0)
        for speed in speed_vals:
            uav = UAV(name="FY1", position0=fy1_start, speed=speed, heading_dir_xy=dir_xy)
            for rt in release_times:
                for ft in fuse_times:
                    explode_time, _, explode_center = compute_explosion_from_release(uav, rt, ft)
                    smoke = SmokeEvent(explode_time=explode_time, explode_center=explode_center)
                    score = compute_shielding_time(m1, TRUE_TARGET_CENTER, [smoke])
                    if score > best.get("score", -1.0):
                        best = {
                            "score": score,
                            "heading_deg": angle_deg,
                            "speed": speed,
                            "release_time": rt,
                            "fuse_time": ft,
                            "explode_time": explode_time,
                            "explode_center_x": explode_center[0],
                            "explode_center_y": explode_center[1],
                            "explode_center_z": explode_center[2],
                        }
    return best


def plan_three_smokes_fy1_for_m1() -> List[Dict[str, float]]:
    # Simple heuristic: place smokes to maintain coverage over 60s horizon, spacing 20s
    # Choose heading approximately toward dummy target for simplicity; then tune release and fuse to hit LoS midpoints
    fy1_start = UAVS_DEFAULT["FY1"]
    heading = heading_towards_xy(fy1_start, (0.0, 0.0))
    # Try speeds 90, 110, 130 and take best sequentially greedily
    candidate_speeds = [90.0, 110.0, 130.0]
    m1 = MISSILES_DEFAULT["M1"]
    best_plan: List[Dict[str, float]] = []
    best_score = -1.0
    for speed in candidate_speeds:
        uav = UAV(name="FY1", position0=fy1_start, speed=speed, heading_dir_xy=heading)
        smokes: List[SmokeEvent] = []
        steps: List[Dict[str, float]] = []
        current_time = 0.0
        # Greedy over 3 smokes
        for i in range(3):
            best_local = None
            best_local_score = -1.0
            # ensure at least 1s apart
            release_candidates = [current_time + 0.0, current_time + 1.0, current_time + 2.0]
            fuse_candidates = [2.0, 2.5, 3.0, 3.5, 4.0]
            for rt in release_candidates:
                for ft in fuse_candidates:
                    explode_time, _, explode_center = compute_explosion_from_release(uav, rt, ft)
                    candidate_smokes = smokes + [SmokeEvent(explode_time=explode_time, explode_center=explode_center)]
                    score = compute_shielding_time(m1, TRUE_TARGET_CENTER, candidate_smokes)
                    if score > best_local_score:
                        best_local_score = score
                        best_local = {
                            "heading_deg": heading_angle_deg_from_dir(heading),
                            "speed": speed,
                            "release_time": rt,
                            "fuse_time": ft,
                            "explode_time": explode_time,
                            "explode_center_x": explode_center[0],
                            "explode_center_y": explode_center[1],
                            "explode_center_z": explode_center[2],
                        }
            assert best_local is not None
            smokes.append(SmokeEvent(explode_time=best_local["explode_time"], explode_center=(best_local["explode_center_x"], best_local["explode_center_y"], best_local["explode_center_z"])) )
            steps.append(best_local)
            current_time = best_local["release_time"] + 1.0
        score_total = compute_shielding_time(m1, TRUE_TARGET_CENTER, smokes)
        if score_total > best_score:
            best_score = score_total
            best_plan = steps
    return best_plan


def write_result1_xlsx(plan: List[Dict[str, float]], out_path: str = "/workspace/result1.xlsx") -> None:
    # Columns: UAV, heading_deg, speed, release_time, fuse_time, explode_time, explode_center_x,y,z
    try:
        import openpyxl  # type: ignore
    except ImportError:
        # Fallback to csv if openpyxl not installed
        import csv
        with open(out_path.replace('.xlsx', '.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["UAV","heading_deg","speed","release_time","fuse_time","explode_time","explode_center_x","explode_center_y","explode_center_z"])
            for step in plan:
                writer.writerow(["FY1", step["heading_deg"], step["speed"], step["release_time"], step["fuse_time"], step["explode_time"], step["explode_center_x"], step["explode_center_y"], step["explode_center_z"]])
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "result1"
    ws.append(["UAV","heading_deg","speed","release_time","fuse_time","explode_time","explode_center_x","explode_center_y","explode_center_z"])
    for step in plan:
        ws.append(["FY1", step["heading_deg"], step["speed"], step["release_time"], step["fuse_time"], step["explode_time"], step["explode_center_x"], step["explode_center_y"], step["explode_center_z"]])
    wb.save(out_path)


def plan_three_uavs_one_smoke_for_m1() -> List[Dict[str, float]]:
    # FY1,FY2,FY3 each one smoke; greedy sequential planning with small grids
    uavs = [
        ("FY1", UAVS_DEFAULT["FY1"]),
        ("FY2", UAVS_DEFAULT["FY2"]),
        ("FY3", UAVS_DEFAULT["FY3"]),
    ]
    m1 = MISSILES_DEFAULT["M1"]
    best_steps: List[Dict[str, float]] = []
    smokes: List[SmokeEvent] = []
    for name, start in uavs:
        best_local = None
        best_local_score = -1.0
        for k in range(36):
            angle_deg = 360.0 * k / 36
            angle_rad = math.radians(angle_deg)
            dir_xy = (math.cos(angle_rad), math.sin(angle_rad), 0.0)
            for speed in [80.0, 100.0, 120.0, 140.0]:
                uav = UAV(name=name, position0=start, speed=speed, heading_dir_xy=dir_xy)
                for rt in [0.0, 0.5, 1.0, 1.5, 2.0]:
                    for ft in [2.0, 2.5, 3.0, 3.5, 4.0]:
                        explode_time, _, explode_center = compute_explosion_from_release(uav, rt, ft)
                        candidate_smokes = smokes + [SmokeEvent(explode_time=explode_time, explode_center=explode_center)]
                        score = compute_shielding_time(m1, TRUE_TARGET_CENTER, candidate_smokes)
                        if score > best_local_score:
                            best_local_score = score
                            best_local = {
                                "UAV": name,
                                "heading_deg": angle_deg,
                                "speed": speed,
                                "release_time": rt,
                                "fuse_time": ft,
                                "explode_time": explode_time,
                                "explode_center_x": explode_center[0],
                                "explode_center_y": explode_center[1],
                                "explode_center_z": explode_center[2],
                            }
        assert best_local is not None
        best_steps.append(best_local)
        smokes.append(SmokeEvent(explode_time=best_local["explode_time"], explode_center=(best_local["explode_center_x"], best_local["explode_center_y"], best_local["explode_center_z"])) )
    return best_steps


def write_result2_xlsx(plan: List[Dict[str, float]], out_path: str = "/workspace/result2.xlsx") -> None:
    try:
        import openpyxl  # type: ignore
    except ImportError:
        import csv
        with open(out_path.replace('.xlsx', '.csv'), 'w', newline='') as f:
            import csv as _csv
            writer = _csv.writer(f)
            writer.writerow(["UAV","heading_deg","speed","release_time","fuse_time","explode_time","explode_center_x","explode_center_y","explode_center_z"])
            for step in plan:
                writer.writerow([step["UAV"], step["heading_deg"], step["speed"], step["release_time"], step["fuse_time"], step["explode_time"], step["explode_center_x"], step["explode_center_y"], step["explode_center_z"]])
        return
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "result2"
    ws.append(["UAV","heading_deg","speed","release_time","fuse_time","explode_time","explode_center_x","explode_center_y","explode_center_z"])
    for step in plan:
        ws.append([step["UAV"], step["heading_deg"], step["speed"], step["release_time"], step["fuse_time"], step["explode_time"], step["explode_center_x"], step["explode_center_y"], step["explode_center_z"]])
    wb.save(out_path)



if __name__ == "__main__":
    val = q1_compute_shield_time()
    print(f"Q1 shield time (s): {val:.2f}")

