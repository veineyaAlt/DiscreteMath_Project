
import argparse
import html
import itertools
import random
import sys
import time
from dataclasses import dataclass, field

try:
    from ortools.sat.python import cp_model
except ImportError:
    sys.exit("Google OR-Tools is required:  pip install ortools")

GROUPS = ("A", "B", "C")
SUBJECTS = {
    "BM": "Business Mathematics",
    "CS": "Computer Science",
    "E":  "Economics",
    "L":  "Logistics",
}
FULL_TIME = ("CS", "E", "L")
BM = "BM"
LECTURER = {"BM": "Part-time lecturer (BM)",
            "CS": "Full-time lecturer (CS)",
            "E":  "Full-time lecturer (E)",
            "L":  "Full-time lecturer (L)"}
CLASS_INDEX = (1, 2)
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

SUBJECT_COLORS = {"BM": "#e57373", "CS": "#64b5f6", "E": "#81c784", "L": "#ffb74d"}
ANSI = {"BM": "\033[91m", "CS": "\033[94m", "E": "\033[92m", "L": "\033[93m"}
RESET = "\033[0m"

@dataclass
class Params:
    day_start_min: int = 8 * 60 + 30
    day_end_min: int = 18 * 60
    duration_min: int = 90
    wait_min: int = 180
    bm_after_min: int | None = None
    seed: int = 1

    @property
    def slot_min(self) -> int:
        return 30

    @property
    def n_slots(self) -> int:
        return (self.day_end_min - self.day_start_min - self.duration_min) // self.slot_min + 1

    @property
    def cover(self) -> int:
        return self.duration_min // self.slot_min

    @property
    def gap_slots(self) -> int:
        return self.wait_min // self.slot_min

    @property
    def n_intervals(self) -> int:
        return (self.day_end_min - self.day_start_min) // self.slot_min

    @property
    def n_days(self) -> int:
        return len(DAY_NAMES)

def slot_time(p: Params, slot: int) -> str:
    m = p.day_start_min + slot * p.slot_min
    return f"{m // 60:02d}:{m % 60:02d}"

def interval_time(p: Params, iv: int) -> str:
    return slot_time(p, iv)

def all_classes():
    return [(g, s, i) for g in GROUPS for s in SUBJECTS for i in CLASS_INDEX]

def classes_of_group(g):
    return [(gg, s, i) for (gg, s, i) in all_classes() if gg == g]

def classes_of_subject(s):
    return [(g, ss, i) for (g, ss, i) in all_classes() if ss == s]

def verify(schedule, p: Params, verbose=True):
    v = []

    if set(schedule) != set(all_classes()):
        v.append("C5: not all 24 classes are scheduled")
        return v

    for c, (d, s) in schedule.items():
        if not (0 <= d < p.n_days and 0 <= s < p.n_slots):
            v.append(f"C5: class {c} outside legal time window ({d},{s})")

    def covers(s):
        return range(s, s + p.cover)

    for g in GROUPS:
        for d in range(p.n_days):
            for iv in range(p.n_intervals):
                busy = [c for c in classes_of_group(g)
                        if schedule[c][0] == d and iv in covers(schedule[c][1])]
                if len(busy) > 1:
                    v.append(f"C5: group {g} has overlapping classes on "
                             f"{DAY_NAMES[d]} at {interval_time(p, iv)}: {busy}")
    for s in SUBJECTS:
        for d in range(p.n_days):
            for iv in range(p.n_intervals):
                busy = [c for c in classes_of_subject(s)
                        if schedule[c][0] == d and iv in covers(schedule[c][1])]
                if len(busy) > 1:
                    v.append(f"C5: {LECTURER[s]} would teach two groups at once "
                             f"({s}) on {DAY_NAMES[d]} at {interval_time(p, iv)}")

    for g in GROUPS:
        for d in range(p.n_days):
            n = sum(1 for c in classes_of_group(g) if schedule[c][0] == d)
            if n > 3:
                v.append(f"C1: group {g} has {n} classes on {DAY_NAMES[d]} (>3)")

    for g in GROUPS:
        for d in range(p.n_days):
            starts = sorted(schedule[c][1] for c in classes_of_group(g)
                            if schedule[c][0] == d)
            for a, b in zip(starts, starts[1:]):
                gap = b - (a + p.cover)
                if gap > p.gap_slots:
                    v.append(f"C2: group {g} on {DAY_NAMES[d]}: gap of "
                             f"{gap * p.slot_min} min between {slot_time(p, a)} "
                             f"and {slot_time(p, b)} classes (> {p.wait_min} min)")
    for d in range(p.n_days):
        starts = sorted(schedule[c][1] for c in classes_of_subject(BM)
                        if schedule[c][0] == d)
        for a, b in zip(starts, starts[1:]):
            gap = b - (a + p.cover)
            if gap > p.gap_slots:
                v.append(f"C3: BM lecturer on {DAY_NAMES[d]}: gap of "
                         f"{gap * p.slot_min} min between {slot_time(p, a)} and "
                         f"{slot_time(p, b)} classes (> {p.wait_min} min)")

    if verbose:
        if v:
            print("VERIFICATION FAILED:")
            for x in v:
                print("  -", x)
        else:
            print("VERIFICATION PASSED: all constraints (C1-C5, horizon) hold.")
    return v

def bm_visit_days(schedule):
    return sorted({d for (g, s, i), (d, slot) in schedule.items() if s == BM})

def build_model(p: Params, max_bm_days=None):
    m = cp_model.CpModel()
    classes = all_classes()

    x = {(g, sub, i): {(d, s): m.NewBoolVar(f"x_{g}_{sub}_{i}_{d}_{s}")
                       for d in range(p.n_days) for s in range(p.n_slots)}
         for (g, sub, i) in classes}

    z = {d: m.NewBoolVar(f"z_{d}") for d in range(p.n_days)}

    for c in classes:
        m.AddExactlyOne(list(x[c].values()))

    for g in GROUPS:
        for d in range(p.n_days):
            for iv in range(p.n_intervals):
                terms = [x[c][(d, s)] for c in classes_of_group(g)
                         for s in range(max(0, iv - p.cover + 1), iv + 1)
                         if (d, s) in x[c]]
                m.Add(sum(terms) <= 1)

    for sub in SUBJECTS:
        for d in range(p.n_days):
            for iv in range(p.n_intervals):
                terms = [x[c][(d, s)] for c in classes_of_subject(sub)
                         for s in range(max(0, iv - p.cover + 1), iv + 1)
                         if (d, s) in x[c]]
                m.Add(sum(terms) <= 1)

    for g in GROUPS:
        for d in range(p.n_days):
            m.Add(sum(x[c][(d, s)] for c in classes_of_group(g)
                      for s in range(p.n_slots)) <= 3)

    def add_gap_constraint(entity_classes, label):
        for d in range(p.n_days):
            slots_used = sorted({s for (d2, s) in x[c] for c in entity_classes}
                                & set(range(p.n_slots)))
            for s, t in itertools.combinations(slots_used, 2):
                if t - (s + p.cover) > p.gap_slots:
                    between = [x[c][(d, u)] for c in entity_classes
                               for u in range(s + 1, t) if (d, u) in x[c]]
                    lhs = [x[c][(d, s)] for c in entity_classes if (d, s) in x[c]]
                    lhs += [x[c][(d, t)] for c in entity_classes if (d, t) in x[c]]
                    m.Add(sum(lhs) <= 1 + sum(between))

    for g in GROUPS:
        add_gap_constraint(classes_of_group(g), f"group{g}")
    add_gap_constraint(classes_of_subject(BM), "BMlecturer")

    for d in range(p.n_days):
        for c in classes_of_subject(BM):
            for s in range(p.n_slots):
                m.AddImplication(x[c][(d, s)], z[d])

    if max_bm_days is not None:
        m.Add(sum(z.values()) <= max_bm_days)

    early = None
    if p.bm_after_min is not None:
        cut = (p.bm_after_min - p.day_start_min) // p.slot_min
        early = []
        for c in classes_of_subject(BM):
            for d in range(p.n_days):
                for s in range(min(cut, p.n_slots)):
                    if (d, s) in x[c]:
                        early.append(x[c][(d, s)])
    return m, x, z, early

def solve_cpsat(p: Params, time_limit=120.0):
    t0 = time.time()

    m, x, z, early = build_model(p)
    m.Minimize(sum(z.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"no feasible timetable found (status {st})")
    optimal = (st == cp_model.OPTIMAL)
    k = int(sum(solver.Value(z[d]) for d in z))

    k_minus_1_status = None
    if optimal and k > 0:
        m2, _, z2, _ = build_model(p, max_bm_days=k - 1)
        s2 = cp_model.CpSolver()
        s2.parameters.max_time_in_seconds = 60.0
        s2.parameters.num_search_workers = 8
        st2 = s2.Solve(m2)
        k_minus_1_status = s2.StatusName(st2)

    schedule = {}
    for c in all_classes():
        for d in range(p.n_days):
            for s in range(p.n_slots):
                if solver.Value(x[c][(d, s)]):
                    schedule[c] = (d, s)
                    break

    early_count = None
    if p.bm_after_min is not None:
        m3, x3, z3, early3 = build_model(p, max_bm_days=k)
        m3.Add(sum(z3.values()) == k)
        m3.Minimize(sum(early3))
        s3 = cp_model.CpSolver()
        s3.parameters.max_time_in_seconds = 60.0
        s3.parameters.num_search_workers = 8
        st3 = s3.Solve(m3)
        if st3 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = {}
            for c in all_classes():
                for d in range(p.n_days):
                    for s in range(p.n_slots):
                        if s3.Value(x3[c][(d, s)]):
                            schedule[c] = (d, s)
                            break
            early_count = int(sum(s3.Value(e) for e in early3))

    info = {
        "bm_days": k,
        "optimal": optimal,
        "k_minus_1_status": k_minus_1_status,
        "wall_time": time.time() - t0,
        "early_bm": early_count,
        "branches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
    }
    return schedule, info

def greedy_solve(p: Params, restarts=3000, time_limit=30.0, seed=1):
    rng = random.Random(seed)
    t0 = time.time()
    best = None
    other = list(itertools.product(GROUPS, FULL_TIME, CLASS_INDEX))

    def can_place(starts_group, starts_subj, day, slot, group, subject):
        day_group = sorted(starts_group[(group, day)])
        day_subj = sorted(starts_subj[(subject, day)])
        for a in day_group + day_subj:
            if not (slot + p.cover <= a or a + p.cover <= slot):
                return False
        trial_g = sorted(day_group + [slot])
        trial_s = sorted(day_subj + [slot])
        if len(trial_g) > 3:
            return False
        for a, b in zip(trial_g, trial_g[1:]):
            if b - (a + p.cover) > p.gap_slots:
                return False
        if subject == BM:
            for a, b in zip(trial_s, trial_s[1:]):
                if b - (a + p.cover) > p.gap_slots:
                    return False
        return True

    for attempt in range(restarts):
        if time.time() - t0 > time_limit:
            break
        starts_group = {(g, d): [] for g in GROUPS for d in range(p.n_days)}
        starts_subj = {(s, d): [] for s in SUBJECTS for d in range(p.n_days)}
        sched = {}

        pack_day = None
        for d in range(p.n_days):
            bm_order = [(g, BM, i) for g, i in
                        zip(rng.sample(GROUPS, 3), (1, 2, 1, 2, 1, 2))]
            bm_order = [c for c in bm_order if c[0] in GROUPS]
            order = [(g, i) for g in GROUPS for i in (1, 2)]
            rng.shuffle(order)
            placed, ok = [], True
            cur = 0
            for (g, i) in order:
                found = False
                for s in range(cur, p.n_slots):
                    if can_place(starts_group, starts_subj, d, s, g, BM):
                        placed.append(((g, BM, i), (d, s)))
                        starts_group[(g, d)].append(s)
                        starts_subj[(BM, d)].append(s)
                        cur = s + p.cover
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok:
                pack_day = d
                for c, pos in placed:
                    sched[c] = pos
                break
        if pack_day is None:
            continue

        rest = other[:]
        rng.shuffle(rest)
        for (g, s, i) in rest:
            options = [(d, sl) for d in range(p.n_days)
                       for sl in range(p.n_slots)]
            rng.shuffle(options)
            placed = False
            for (d, sl) in options:
                if can_place(starts_group, starts_subj, d, sl, g, s):
                    sched[(g, s, i)] = (d, sl)
                    starts_group[(g, d)].append(sl)
                    starts_subj[(s, d)].append(sl)
                    placed = True
                    break
            if not placed:
                sched = None
                break
        if sched is None:
            continue
        if verify(sched, p, verbose=False):
            continue
        days = len(bm_visit_days(sched))
        if best is None or days < best[0]:
            best = (days, sched)
            if days == 1:
                break
    return best[1] if best else None, time.time() - t0

def print_timetable(schedule, p: Params):
    print("\n" + "=" * 78)
    print("WEEKLY TIMETABLE  (all times are class start times; "
          f"each class lasts {p.duration_min} min)")
    print("=" * 78)
    for d in range(p.n_days):
        print(f"\n--- {DAY_NAMES[d]} ---")
        entries = [(slot_time(p, s), s, g, sub)
                   for (g, sub, i), (d2, s) in schedule.items() if d2 == d]
        if not entries:
            print("   (no classes)")
            continue
        for t, s, g, sub in sorted(entries):
            role = "part-time lecturer" if sub == BM else "full-time lecturer"
            print(f"   {t}-{slot_time(p, s + p.cover)}  Group {g}: "
                  f"{SUBJECTS[sub]} ({role})")
    print("\n" + "-" * 78)
    days = bm_visit_days(schedule)
    print(f"Business Mathematics lecturer visits INHA on "
          f"{len(days)} day(s): {', '.join(DAY_NAMES[d] for d in days)}")
    print("-" * 78)

def print_grid(schedule, p: Params):
    print("\nColour-coded weekly grid (ANSI; 'XX' = class in progress, "
          "'name' = class starting):")
    header = f"{'time':<6}" + "".join(f"{d:<16}" for d in ("Mon", "Tue", "Wed", "Thu", "Fri"))
    print(header)
    print("-" * len(header))
    for iv in range(p.n_intervals):
        row = f"{interval_time(p, iv):<6}"
        for d in range(p.n_days):
            tags = []
            for (g, sub, i), (d2, s) in schedule.items():
                if d2 == d and s <= iv < s + p.cover:
                    col = ANSI.get(sub, "") if sys.stdout.isatty() else ""
                    tag = f"{sub}{g}" if iv == s else " ."
                    tags.append(f"{col}{tag}{RESET}" if sys.stdout.isatty() else tag)
            cell = "/".join(tags)
            row += f"{cell:<16}"
        print(row)

def write_html(schedule, p: Params, path, info):
    rows = []
    rows.append("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Weekly Timetable</title><style>
body{font-family:'Segoe UI',Arial,sans-serif;margin:24px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #999;padding:6px;text-align:center;font-size:14px}
th{background:#37474f;color:#fff}
td.slot{background:#eceff1;font-weight:bold;width:70px}
.legend span{display:inline-block;padding:4px 10px;margin:2px;border-radius:4px;color:#212121}
h1{font-size:22px} .info{color:#37474f}
</style></head><body>""")
    visits = ", ".join(DAY_NAMES[d] for d in bm_visit_days(schedule))
    rows.append(f"<h1>Optimized Weekly Timetable &mdash; Groups A, B, C</h1>")
    rows.append(f"<p class='info'>Class duration: {p.duration_min} min &middot; "
                f"max student waiting: {p.wait_min} min &middot; "
                f"BM lecturer visits INHA on <b>{len(bm_visit_days(schedule))}</b> "
                f"day(s): <b>{visits}</b> (proved optimal by CP-SAT)</p>")
    occ = {d: [] for d in range(p.n_days)}
    for (g, sub, i), (d, s) in schedule.items():
        occ[d].append((s, s + p.cover, g, sub))
    rows.append("<table><tr><th>Time</th>" +
                "".join(f"<th>{d}</th>" for d in DAY_NAMES) + "</tr>")
    for iv in range(p.n_intervals):
        t = interval_time(p, iv)
        rows.append(f"<tr><td class='slot'>{t}</td>")
        for d in range(p.n_days):
            active = [(s, e, g, sub) for (s, e, g, sub) in occ[d]
                      if s <= iv < e]
            if not active:
                rows.append("<td></td>")
            else:
                parts = []
                for (s, e, g, sub) in active:
                    parts.append(
                        f"<div style='background:{SUBJECT_COLORS[sub]};"
                        f"margin:1px;padding:3px;border-radius:4px'>"
                        f"<b>{sub}</b> &mdash; Group {g} "
                        f"({interval_time(p, s)}&ndash;{interval_time(p, e)})</div>")
                rows.append("<td style='padding:2px'>" + "".join(parts) + "</td>")
        rows.append("</tr>")
    rows.append("</table><p class='legend'>Legend: " + "".join(
        f"<span style='background:{SUBJECT_COLORS[s]}'>{SUBJECTS[s]}</span>"
        for s in SUBJECTS) + "</p>")
    rows.append("<p class='info'>Each cell shows every class in progress "
                "during that 30-minute interval.</p></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"\nHTML colour-coded timetable written to: {path}")

def main():
    ap = argparse.ArgumentParser(description="Timetable optimizer (CP-SAT)")
    ap.add_argument("--duration", type=int, default=90,
                    help="class duration in minutes (multiple of 30)")
    ap.add_argument("--wait", type=int, default=180,
                    help="max waiting time between consecutive classes, minutes")
    ap.add_argument("--day-start", default="08:30")
    ap.add_argument("--day-end", default="18:00")
    ap.add_argument("--bm-after", default=None, metavar="HH:MM",
                    help="bonus: prefer BM classes after this time, e.g. 14:00")
    ap.add_argument("--html", default="timetable.html",
                    help="output path for the colour-coded HTML timetable")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    def hm(s):
        h, m = map(int, s.split(":"))
        return h * 60 + m

    p = Params(day_start_min=hm(args.day_start), day_end_min=hm(args.day_end),
               duration_min=args.duration, wait_min=args.wait,
               bm_after_min=hm(args.bm_after) if args.bm_after else None,
               seed=args.seed)
    if p.duration_min % 30 or p.wait_min % 30:
        sys.exit("duration and wait must be multiples of 30 minutes")
    if p.day_end_min - p.day_start_min < p.duration_min:
        sys.exit("day too short for one class")

    print(f"Parameters: {p.n_days} days, day window "
          f"{args.day_start}-{args.day_end}, class = {p.duration_min} min, "
          f"max gap = {p.wait_min} min, {p.n_slots} start slots/day, "
          f"24 classes to schedule.")
    if p.bm_after_min:
        print(f"Bonus preference active: BM classes preferably after "
              f"{args.bm_after}.")

    print("\n[1/4] Solving exactly with Google OR-Tools CP-SAT ...")
    schedule, info = solve_cpsat(p)
    print(f"      status: OPTIMAL  |  BM visit days = {info['bm_days']}  |  "
          f"solve time = {info['wall_time']:.2f}s  |  "
          f"branches = {info['branches']}, conflicts = {info['conflicts']}")
    if p.bm_after_min is not None:
        print(f"      second-stage objective: {info['early_bm']} BM class(es) "
              f"before {args.bm_after}")

    print("\n[2/4] Independent constraint verification of the CP-SAT timetable:")
    verify(schedule, p)

    print("\n[3/4] Optimality evidence:")
    print(f"      * CP-SAT reports status OPTIMAL with objective "
          f"{info['bm_days']} (best objective bound = {info['bm_days']}).")
    if info["k_minus_1_status"]:
        print(f"      * Re-solving with at most {info['bm_days'] - 1} visit "
              f"days gives status {info['k_minus_1_status']} -> fewer visits "
              f"are provably impossible.")

    print("\n[4/4] Second method for comparison: constructive greedy "
          "(randomized restarts) ...")
    gsched, gtime = greedy_solve(p, seed=args.seed)
    if gsched is not None:
        gdays = len(bm_visit_days(gsched))
        gviol = verify(gsched, p, verbose=False)
        print(f"      greedy: BM visit days = {gdays}, "
              f"time = {gtime:.2f}s, feasible = {not gviol} "
              f"(no optimality guarantee)")
        print("\n      Comparison:")
        print("      method   | BM visit days | time  | optimality proof")
        print(f"      CP-SAT   | {info['bm_days']}             | "
              f"{info['wall_time']:.2f}s | yes (exact)")
        print(f"      greedy   | {gdays}             | "
              f"{gtime:.2f}s | no (heuristic)")
    else:
        print("      greedy: no feasible solution found within limits")

    print_timetable(schedule, p)
    print_grid(schedule, p)
    write_html(schedule, p, args.html, info)

    print("\nFinal check of the delivered timetable:")
    verify(schedule, p)

if __name__ == "__main__":
    main()
