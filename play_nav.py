"""Walk Myst as the player -- the navigation/inventory layer, interactive.

This drives the PLAYER model (nav-myst), not the world-state game (that's
play.py). One Ego moves card-to-card across the island and the five ages. The
puzzles are host seams (PuzzleStub, autoSolve=1): attempting one here just
solves it. The library tower is the access gate -- run it to LOCATE the four
linking books; an age cannot be attempted until its book is located.

    # generate the Python target first:
    python -m earhart.cli compile nav-myst.kinner.json --lib lib \
           --target python --output-dir bin-nav-py

    python play_nav.py            # interactive
    python play_nav.py auto 80    # random walk, 80 steps (smoke test)

The puzzle/clue plumbing (a stub solving, a book being located) auto-settles,
so you only ever pick real player moves: walk somewhere, run the tower, attempt
a puzzle, take a linking book.

NOTE: nav-myst now ships its PuzzleStubs at autoSolve=0 (they delegate to a
host), so this file's auto-settle no longer SOLVES a puzzle -- attempting one
leaves the stub Located. For the full playable game where the host RUNS each
mini-puzzle (and the imager/fireplace prompts), use play_game.py. This module's
nav helpers (pretty/label_of/_bfs/_walk_adj/moves/settle) are reused there; its
own `demo` (which assumed auto-solve) is superseded by play_game.py's `demo`.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "bin-nav-py"))
from NavMyst import NavMyst  # noqa: E402

# real authored move labels, keyed by (fromCard, toCard), straight from source
_PL = json.loads((HERE / "lib" / "PlayerLocation.kinner.json").read_text("utf-8"))
EVENT = {(t["currentState"], t["newState"]): t["event"]
         for t in _PL["actors"][0]["triples"]}
# the five clue cards: four ObservCode* read off the rotating tower, plus the
# fireplace pattern (Dunny). Reading one records the clue that age's puzzle needs.
CLUE_CARDS = {"Mechanical": "ObservCodeGearC864", "Stoneship": "ObservCodeBoatC847",
              "Channelwood": "ObservCodeTreeC867", "Selenitic": "ObservCodeSpaceC866",
              "Dunny": "FirePlace1SWDoorUpC871"}
# the eight page Item aliases (Sirrus = red, Achenar = blue; one per age)
PAGE_NAMES = [b + a for b in ("Sirrus", "Achenar")
              for a in ("Mechanical", "Stoneship", "Channelwood", "Selenitic")]


def pretty(card: str) -> str:
    """'Boardwalk36SC87' -> 'Boardwalk 36 S (#87)'."""
    m = re.search(r"C(\d+)$", card)
    num = m.group(1) if m else "?"
    core = re.sub(r"C\d+$", "", card)
    core = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", core)
    core = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", core)
    return f"{core} (#{num})"


def label_of(t) -> str:
    """Readable label from the (unique) transition name -- not (from,to), which
    collides for the four same-card deposit self-loops."""
    nm = t.name[4:] if t.name.startswith("Ego_") else t.name
    if t.from_state != t.to_state and nm.startswith("WalksTo"):
        return "go to " + pretty(t.to_state)
    if nm.endswith("_" + t.from_state):           # strip the from-state disambiguator
        nm = nm[: -(len(t.from_state) + 1)]
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", nm)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_player(name: str) -> bool:
    return name.startswith("PL.")


def settle(app) -> None:
    """Auto-fire the non-player plumbing (a stub going Located->Solved, a Clue
    being located) -- any non-PL transition that actually changes a state."""
    for _ in range(500):
        pending = [n for n, _, t in app._enabled()
                   if not is_player(n) and t.from_state != t.to_state]
        if not pending:
            return
        app.fire(pending[0])


def moves(app):
    """Real player choices: PL transitions that change something (a walk, or a
    self-loop that sends -- attempt / run-the-tower)."""
    out = []
    for n, c, t in app._enabled():
        if not is_player(n):
            continue
        if t.from_state == t.to_state and not t.sends:
            continue  # idle self-loop
        out.append((n, label_of(t), t))
    return out


def status(app) -> str:
    c = app._components
    here = c["PL"].state
    cluesread = [age for age in CLUE_CARDS if c["Clue" + age].state == "Seen"]
    entered = [age for age, comp in (("Mechanical", "MechanicalPuz"),
               ("Stoneship", "StoneshipPuz"), ("Channelwood", "ChannelwoodPuz"),
               ("Selenitic", "SeleniticPuz"), ("Dunny", "DunnyPuz"))
               if c[comp].state == "Solved"]
    see = ""
    for age, card in CLUE_CARDS.items():
        if here == card:
            see = f"\n  You see the {age} clue here in the tower. (read it)"
    carrying = next((n for n in PAGE_NAMES if c["Page" + n].state == "Held"), None)
    deposited = sum(1 for n in PAGE_NAMES if c["Page" + n].state == "Gone")
    hands = "carrying " + carrying if carrying else "empty-handed"
    return (f"\n== You are at {pretty(here)} =={see}\n"
            f"  clues read: {', '.join(cluesread) or '(none)'}"
            f"   |   ages solved: {', '.join(entered) or '(none)'}\n"
            f"  hands: {hands}   |   pages delivered: {deposited}/8")


def _walk_adj():
    # navigable edges for pathfinding: ordinary walks plus the ungated link-back
    # (an age's return book -> the library), so a route can leave an age.
    adj = {}
    for (a, b), ev in EVENT.items():
        if ev.startswith("walks to") or ev.startswith("take the Myst linking book back"):
            adj.setdefault(a, []).append(b)
    return adj


def _bfs(adj, start, goal):
    seen, q = {start: None}, [start]
    while q:
        x = q.pop(0)
        if x == goal:
            path = [x]
            while seen[path[-1]] is not None:
                path.append(seen[path[-1]])
            return path[::-1]
        for y in adj.get(x, []):
            if y not in seen:
                seen[y] = x
                q.append(y)
    return None


def _fire_to(app, to_card):
    """Fire the enabled player move that lands on to_card."""
    for n, _, t in app._enabled():
        if is_player(n) and t.to_state == to_card and t.from_state != t.to_state:
            app.fire(n)
            return True
    return False


def _fire_action(app, needle):
    nd = _norm(needle)
    for n, label, t in moves(app):
        if nd in _norm(label):
            app.fire(n)
            return True
    return False


def demo(app) -> int:
    """Autopilot the real journey into Mechanical: ride the elevator up the
    tower to the Gear clue card and read it, walk to the gear gate, attempt the
    puzzle, take the linking book."""
    adj = _walk_adj()
    legs = [(CLUE_CARDS["Mechanical"], "read the Mechanical clue"),
            ("GearGate2C840", "attempt the Mechanical")]
    for target, action in legs:
        here = app._components["PL"].state
        path = _bfs(adj, here, target)
        if not path:
            print(f"!! no walk path {here} -> {target}")
            return 1
        for nxt in path[1:]:
            assert _fire_to(app, nxt), f"could not step to {nxt}"
            settle(app)
        print(status(app))
        print(f"  >>> {action}")
        assert _fire_action(app, action), f"action unavailable: {action}"
        settle(app)
    print(status(app))
    assert _fire_action(app, "take the Mechanical"), "linking book not takeable"
    settle(app)
    print(status(app))
    print(f"  >>> entered Mechanical: now at {pretty(app._components['PL'].state)}")

    # inventory leg: grab Sirrus's red page in Mechanical, carry it back to the
    # library, and deposit it in Sirrus's red book.
    inv = [("SCacheChestC455", "pick up Sirrus's red page"),
           ("Library2NC783", "place the Mechanical page in Sirrus's book")]
    for target, action in inv:
        here = app._components["PL"].state
        path = _bfs(adj, here, target)
        if not path:
            print(f"!! no walk path {here} -> {target}")
            return 1
        for nxt in path[1:]:
            assert _fire_to(app, nxt), f"could not step to {nxt}"
            settle(app)
        print(status(app))
        print(f"  >>> {action}")
        assert _fire_action(app, action), f"action unavailable: {action}"
        settle(app)
    print(status(app))
    print("\n  >>> delivered Sirrus's Mechanical page.")
    return 0


def main() -> int:
    app = NavMyst()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        return demo(app)
    auto = len(sys.argv) > 1 and sys.argv[1] == "auto"
    cap = int(sys.argv[2]) if auto and len(sys.argv) > 2 else 60
    print(__doc__.splitlines()[0])
    n = 0
    while True:
        settle(app)
        opts = moves(app)
        if not opts:
            print("\nNo moves left -- stuck.")
            return 1
        if auto:
            import random
            name, label, _ = random.Random(n).choice(opts)
            if n % 10 == 0 or app._components["PL"].state in CLUE_CARDS.values():
                print(status(app))
            print(f"[{n:3}] {label}")
            app.fire(name)
            n += 1
            if n >= cap:
                print(f"\n(auto stopped at {cap} moves)")
                return 0
            continue
        print(status(app))
        for i, (_, label, _t) in enumerate(opts):
            print(f"  {i:2}. {label}")
        raw = input("move (number / q): ").strip()
        if raw in ("q", "quit"):
            return 0
        if not raw.isdigit() or int(raw) >= len(opts):
            print("  ?")
            continue
        app.fire(opts[int(raw)][0])
        n += 1


if __name__ == "__main__":
    raise SystemExit(main())
