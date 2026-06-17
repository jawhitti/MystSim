"""play_game.py -- the host-driven, fully playable Myst.

This is the runnable game. You walk the real island/age map (PlayerLocation),
and when you ATTEMPT an access puzzle the PuzzleStub -- now autoSolve=0 -- fires
Request at the HOST and waits on RESOLVE. The host RUNS THE ACTUAL MINI-PUZZLE:
its own generated-Python sub-game (clock-tower gears, planetarium, boiler,
voltage), the SAME model TLC verified standalone. You solve the sub-game; the
host fires Solved back (app.p.<alias>_resolve), the age unlocks, and play
continues. The imager and the fireplace are "enter the number / pattern" host
prompts -- there is no sub-model to run, per design (the count/pattern is the
glossed advisory clue). So the whole loop is design -> prove -> generate -> play,
all the way down to the mini-puzzles.

    python play_game.py            # interactive: walk + solve
    python play_game.py demo       # self-test: drive every host puzzle (smoke)
    python play_game.py autowin    # win the WHOLE game live, through the host

The host also enforces the assume/guarantee precondition the stub no longer
checks in autoSolve=0 mode: you cannot attempt a puzzle until you have read its
clue (the tower marker for an age; the fireplace combination for Dunny). The
menu hides an attempt until its clue is in hand, so the stub never strands in
the Located state.
"""
import importlib
import os
import re
import sys
from pathlib import Path

# play_nav owns the nav helpers and the compiled NavMyst import (bin-nav-py).
import play_nav as N

HERE = Path(__file__).parent

# move-by-move narration (autowin sets it on): _log numbers each player move
# (walk or action); _say annotates sub-game moves / host notes without a number.
_VERBOSE = False
_STEP = [0]

# ANSI colour-coding for the move narration, by category. Enabled only on a real
# terminal (so a redirected log stays plain text). On Windows, os.system("")
# turns on the console's ANSI/VT processing.
_USE_COLOR = sys.stdout.isatty()
if _USE_COLOR and os.name == "nt":
    os.system("")
_COLORS = {
    "walk": "\033[90m",   # grey  -- walking card to card
    "item": "\033[93m",   # yellow-- picked up an item (a page)
    "clue": "\033[94m",   # blue  -- saw / read a clue
    "book": "\033[92m",   # green -- used a linking book to travel between ages
}
_RESET = "\033[0m"


def _color(s: str, kind: str) -> str:
    c = _COLORS.get(kind, "") if _USE_COLOR else ""
    return f"{c}{s}{_RESET}" if c else s


def _classify(label: str) -> str:
    """Category for a player action label -> colour key (default 'action')."""
    low = label.lower()
    if "pick up" in low:
        return "item"
    if ("clue" in low or "catherine" in low or "fireplace combination" in low
            or "journal" in low):
        return "clue"
    if "linking book" in low:
        return "book"
    return "action"


def _log(msg: str, kind: str = "action") -> None:
    if _VERBOSE:
        _STEP[0] += 1
        print(_color(f"  [{_STEP[0]:4}] {msg}", kind))


def _say(msg: str) -> None:
    if _VERBOSE:
        print(f"          {msg}")


# the eight island marker switches, by alias suffix -- the imager validates the
# player's typed count against how many of these read On right now.
SWITCHES = ["Dock", "Cabin", "Clock", "Gears", "SpaceShip",
            "Generator", "Tree", "Planetarium"]

# stub alias -> (sub-game bin dir, its App class). The four access puzzles run
# their real standalone model to its Open state.
SUBGAME = {
    "MechanicalPuz":  ("bin-puzzles/clock",       "ClockGame"),
    "StoneshipPuz":   ("bin-puzzles/planetarium", "PlanetariumGame"),
    "ChannelwoodPuz": ("bin-puzzles/boiler",      "BoilerGame"),
    "SeleniticPuz":   ("bin-puzzles/voltage",     "VoltageGame"),
}
# stub alias -> the clue component the host requires Seen before it will run.
CLUE_OF = {
    "MechanicalPuz": "ClueMechanical", "StoneshipPuz": "ClueStoneship",
    "ChannelwoodPuz": "ClueChannelwood", "SeleniticPuz": "ClueSelenitic",
    "DunnyPuz": "FireplaceCombo",
}
AGE2STUB = {"Mechanical": "MechanicalPuz", "Stoneship": "StoneshipPuz",
            "Channelwood": "ChannelwoodPuz", "Selenitic": "SeleniticPuz",
            "Dunny": "DunnyPuz"}

# The checker's own verified solution for each sub-game, as ordered
# transition-name substrings. Used by `demo`/auto to drive a sub-game with no
# human at the keyboard (interactive play lets you find the path yourself); this
# is exactly the move sequence the standalone NeverSolved probe produced.
# Each entry is (name-substring to fire, human label). The labels spell out the
# moves -- especially the clock tower's left/right/HOLD lever sequence -- and
# run_subgame prints the puzzle's variable state after each one, so you can watch
# the gear digits move (and see the hold bump ONLY the middle wheel).
SOLUTION = {
    "ClockGame": (
        [("TurnTheLargeWheel", "turn large wheel (minute +5)")] * 8
        + [("TurnTheSmallWheel", "turn small wheel (hour +1)")] * 2
        + [("PressTheRedButton", "press the red button -- the gear bridge rises"),
           ("LeftLeverTap", "pull LEFT lever  (mid+1, bot+1)"),
           ("RightLeverTap", "pull RIGHT lever (top+1, mid+1)"),
           ("RightLeverTap", "pull RIGHT lever (top+1, mid+1)"),
           ("HoldALever", "HOLD a lever     (mid+1 ONLY)"),
           ("HoldALever", "HOLD a lever     (mid+1 ONLY)"),
           ("TheTowerOpens", "the tower opens")]),
    "PlanetariumGame": [
        ("SetTheFirstDate", "set the first date, record its constellation"),
        ("SetTheSecondDate", "set the second date, record its constellation"),
        ("SetTheThirdDate", "set the third date, record its constellation"),
        ("AllThreeConstellations", "all three matched -- go to the birdbath"),
        ("PressTheThreeButtons", "press the three buttons -- the ships rise")],
    "BoilerGame": [
        ("OpenTheWallSafe", "open the wall safe, take the match"),
        ("LightTheMatch", "light the match in the furnace"),
        ("TurnTheRightWheelClockwise", "turn the wheel clockwise until it blocks"),
        ("WaitForTheClunking", "wait for the clunking to cease"),
        ("TurnTheWheelCounterclockwise", "turn counterclockwise -- the tree rises"),
        ("RushOutOfTheCabin", "rush out to the great tree"),
        ("ClimbTheRisingTree", "climb the rising tree to the book")],
    "VoltageGame": [
        ("PressThe22", "press the 22 V button"),
        ("PressThe19", "press the 19 V button"),
        ("PressThe18", "press the 18 V button"),
        ("TheVoltageReadsExactly59", "voltage reads 59 -- the rocket powers up"),
        ("PressKeyboardKey1", "play sound key 1"),
        ("PressKeyboardKey2", "play sound key 2"),
        ("PressKeyboardKey3", "play sound key 3"),
        ("PressKeyboardKey4", "play sound key 4"),
        ("PressKeyboardKey5", "play sound key 5"),
        ("PullTheLever", "pull the lever -- the Selenitic book appears")],
    # Selenitic INTERIOR sub-games (not access stubs -- run as gates on the descent).
    "SoundGame": [
        ("DrippingWater", "light the dripping-water symbol (blue page nearby)"),
        ("FissureSymbol", "light the fissure symbol"),
        ("ClockSymbol", "light the clock symbol"),
        ("CrystalSymbol", "light the crystal symbol (red page on top)"),
        ("TunnelSymbol", "light the tunnel symbol"),
        ("FlipTheTunnelLight", "flip the tunnel light switch"),
        ("AimTheWaterReceiver", "aim the display: water receiver -> its sound"),
        ("AimTheFissureReceiver", "aim the display: fissure receiver -> its sound"),
        ("AimTheClockReceiver", "aim the display: clock receiver -> its sound"),
        ("AimTheCrystalReceiver", "aim the display: crystal receiver -> its sound"),
        ("AimTheTunnelReceiver", "aim the display: tunnel receiver -> its sound"),
        ("PressSigma", "press Sigma -> playback order: crystal,water,tunnel,fissure,clock"),
        ("SetSlider1", "slider 1 = crystal"),
        ("SetSlider2", "slider 2 = water"),
        ("SetSlider3", "slider 3 = tunnel"),
        ("SetSlider4", "slider 4 = fissure"),
        ("SetSlider5", "slider 5 = clock"),
        ("VesselDoorOpens", "press the button -- the vessel door opens")],
    "MazeGame": [(d, f"ride {d}") for d in
                 ["North", "West", "North", "East", "East", "South", "South", "West",
                  "Southwest", "West", "Northwest", "Northeast", "North", "Southeast"]],
    # Channelwood INTERIOR sub-game (gates the return book) -- the windmill/valve
    # network: faucet -> windmill power -> set the four routing valves -> route to
    # the first elevator (upper village) -> re-route to the bridge -> crank the pipe
    # -> re-route to the final elevator -> ride up to the Myst linking book.
    "ChannelwoodGame": [
        ("TurnTheFaucet", "turn the faucet -- water flows from the windmill"),
        ("TheWindmillTurns", "the windmill turns -- the pipes have power"),
        ("SetTheFirstValve", "valve 1: cover the line to the spiral staircase"),
        ("SetTheSecondValve", "valve 2: water flows right"),
        ("SetTheThirdValve", "valve 3: water flows right"),
        ("SetTheFourthValve", "valve 4: water flows left"),
        ("RouteTheWaterToTheFirstElevator", "route the water to the first elevator"),
        ("RideTheFirstElevatorUp", "ride the first elevator up to the upper village"),
        ("PullTheHutLever", "pull the hut lever, descend the powered stairway (the red page is in the desk drawer -- carried out one at a time)"),
        ("RideTheFirstElevatorBackDown", "ride the first elevator back down"),
        ("ReRouteTheWaterToTheBridge", "re-route the water to the bridge machine"),
        ("PullTheBridgeLever", "pull the lever -- the bridge rises -- and cross"),
        ("TurnTheCrank", "turn the crank to extend the final pipe to the far side"),
        ("ReRouteTheWaterToTheFinalElevator", "re-route the water to the final elevator"),
        ("RideTheFinalElevatorUp", "ride the final elevator up to the Myst linking book")],
    # Stoneship INTERIOR sub-game (gates the return book) -- the lighthouse/battery:
    # telescope bearing -> drain the lighthouse + the chest air/refloat/key -> padlock
    # + trapdoor -> crank the generator six turns -> drain the rock -> both dark tunnels
    # -> the SE compass press -> drain the ship deck -> the linking book rises.
    "StoneshipGame": [
        ("ReadTheLighthouseBearing", "read the telescope -- the lighthouse is at 135 degrees"),
        ("PressTheRightUmbrellaButton", "right umbrella button -- drain the lighthouse stairs"),
        ("DescendTheLighthouse", "descend, turn the chest valve to trap its air"),
        ("PressTheRightUmbrellaButtonAgain", "right button again -- refill, the chest floats up"),
        ("OpenTheFloatedChest", "open the chest, take the key"),
        ("UnlockThePadlock", "unlock the padlock with the key"),
        ("OpenTheTrapdoor", "open the trapdoor, climb to the generator"),
        ("CrankTheGenerator", "crank the generator (rotation 1)"),
        ("CrankTheGenerator", "crank the generator (rotation 2)"),
        ("CrankTheGenerator", "crank the generator (rotation 3)"),
        ("CrankTheGenerator", "crank the generator (rotation 4)"),
        ("CrankTheGenerator", "crank the generator (rotation 5)"),
        ("CrankTheGenerator", "crank the generator (rotation 6)"),
        ("TheBatteriesAreFull", "the batteries are full -- the lights have power"),
        ("PressTheMiddleUmbrellaButton", "middle umbrella button -- drain the rock, enter the tunnels"),
        ("GoDownATunnel", "tunnel 1: go down by the battery's light (blue page on Achenar's bed -- carried out one at a time)"),
        ("GoDownATunnel", "tunnel 2: go down by the battery's light (red page in Sirrus's drawer)"),
        ("PressTheCompassRoseSoutheastButton", "compass-rose: press SOUTHEAST (135 degrees)"),
        ("PressTheLeftUmbrellaButton", "left umbrella button -- drain the ship deck, the book rises")],
    "FortressGame": [
        ("EnterTheFortress", "enter the fortress, explore the throne rooms"),
        ("StudyTheImager", "study the imager -- learn the sound->direction key"),
        ("LowerTheHallwayStairs", "lower the hallway stairs + work the lever"),
        ("RideTheElevator", "ride the elevator up"),
        ("PressTheDownButton", "press the DOWN button"),
        ("SlipOutBefore", "slip out before the car drops -- now on the roof controls (the trick!)"),
        ("Position0", "note the symbol at position 0"),
        ("RotateToTheNext", "rotate the fortress"),
        ("Position1", "note the symbol at position 1"),
        ("RotateToTheNext", "rotate the fortress"),
        ("Position2", "note the symbol at position 2"),
        ("RotateToTheNext", "rotate the fortress"),
        ("Position3", "note the symbol at position 3"),
        ("LockTheFortress", "orient the fortress SOUTH (the heading the symbols point to)"),
        ("SetTheFinalPodium", "set the final podium to match the symbols seen at the other positions"),
        ("PressTheRedButton", "press the red button -- the linking-book stairs descend")],
}

# Selenitic interior gates: walking INTO this card first requires the host to run
# these sub-games (bindir, appclass, solved_state). The sound puzzle opens the
# maze-hut door; the maze is then ridden. Tracked in _solved_gates so each fires once.
INTERIOR_GATES = {
    # Selenitic: the maze-hut door -- sound puzzle opens the vessel, then ride the maze.
    "MainPath3NopenC1082": [("bin-puzzles/sound", "SoundGame", "Open"),
                            ("bin-puzzles/maze", "MazeGame", "Book")],
    # Mechanical: the return book -- rotate the fortress (elevator trick + 4 symbols) first.
    "SouthStairs3NOpenC558": [("bin-puzzles/fortress", "FortressGame", "Done")],
    # Channelwood: the return book -- run the windmill/valve network (route the water
    # through the four valves, work the upper village, cross the bridge, crank the pipe)
    # to ride the final elevator up before you can reach the linking book.
    "VillageBookRoom1nBookopenC11": [("bin-puzzles/channelwood", "ChannelwoodGame", "RodeElevator")],
    # Stoneship: the return book -- the lighthouse/battery puzzle (telescope, drain, key,
    # crank, both tunnels, the SE compass press) must reach Recovered before the linking book.
    "BookRoomBookopenC1327": [("bin-puzzles/stoneship", "StoneshipGame", "Recovered")],
}
_solved_gates: set = set()


def settle(app) -> None:
    """Auto-fire the non-player plumbing -- but NOT the host-driven RESOLVE.
    A stub's RESOLVE in-port is bound '@host', so the runtime treats it as
    always-enabled (the environment may deliver anytime); left to N.settle it
    would self-resolve every attempted stub and bypass the host. The host fires
    RESOLVE explicitly (run_host), so we hold it back here."""
    for _ in range(2000):
        pending = [n for n, _, t in app._enabled()
                   if not N.is_player(n) and t.recv_port != "RESOLVE"
                   # state-changers AND message-consuming self-loops (a Book
                   # counting a deposited page is Trapped->Trapped with a
                   # PAGE_IN receive -- skip it and the channel jams), but not
                   # idle no-op loops (StaysSolved: from==to, no receive).
                   and (t.from_state != t.to_state or t.recv_port is not None)]
        if not pending:
            return
        app.fire(pending[0])


def resolve_attr(alias: str) -> str:
    """'MechanicalPuz' -> 'mechanical_puz_resolve' (the open RESOLVE in-port)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", alias).lower() + "_resolve"


def sub_label(t) -> str:
    """Readable label off a sub-game transition name ('SeleniticVoltage_PressThe22VButton'
    -> 'Press The 22 V Button')."""
    nm = t.name.split("_", 1)[1] if "_" in t.name else t.name
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", nm)


def switches_on(app) -> int:
    return sum(app._components[f"Switch{s}"].state == "On" for s in SWITCHES)


def attempt_age(t):
    """If transition t is a 'solve the <Age> access mini-puzzle' attempt, return
    the age; else None. (Matched off the unique transition name.)"""
    if "AccessMiniPuzzle" not in t.name:
        return None
    return next((age for age in AGE2STUB if age in t.name), None)


def is_imager(t) -> bool:
    return "Imager" in t.name


# --------------------------------------------------------------------------- #
# the host: run one mini-puzzle as its own generated-Python sub-game
# --------------------------------------------------------------------------- #

_SUB_CACHE: dict[str, object] = {}


def _sub_app(bindir: str, appclass: str):
    if appclass not in _SUB_CACHE:
        sys.path.insert(0, str(HERE / bindir))
        _SUB_CACHE[appclass] = importlib.import_module(appclass)
    return getattr(_SUB_CACHE[appclass], appclass)


def _pvars(comp) -> str:
    """Compact snapshot of a sub-game component's variables (the puzzle digits)."""
    vs = {k: v for k, v in vars(comp).items()
          if not k.startswith("_")
          and k not in ("state", "initial_state", "state_constants")}
    return " ".join(f"{k}={v}" for k, v in vs.items())


def _match(sub, needle):
    """Find an enabled transition for needle: exact event-name match first (so the
    maze's 'North' doesn't also match 'Northwest'), then substring fallback (for
    needles like 'PressThe22' that are name prefixes)."""
    return (next(((n, t) for n, _, t in sub._enabled() if t.name.split("_", 1)[-1] == needle), None)
            or next(((n, t) for n, _, t in sub._enabled() if needle in t.name), None))


def run_subgame(bindir: str, appclass: str, script=None, solved_state: str = "Open") -> bool:
    """Drive the mini-puzzle's own app to its solved state. script=list of
    (name-substring, label) pairs drives it headless (returns True on solve),
    printing each move and the resulting state; otherwise a little REPL lets the
    player solve it. Reaching the solved state trips the model's NeverSolved canary
    -- we catch that as the win."""
    App = _sub_app(bindir, appclass)
    sub = App()
    P = sub._components["P"]

    if script is not None:
        for needle, label in script:
            if P.state == solved_state:
                break
            hit = _match(sub, needle)
            if not hit:
                print(f"    [auto] move {needle!r} unavailable at state {P.state}")
                return False
            try:
                sub.fire(hit[0])
            except App.InvariantViolation:           # canary realized = solved reached
                _say(f"- {label:<42} -> [{solved_state}] {_pvars(P)}")
                return True
            _say(f"- {label:<42} -> [{P.state}] {_pvars(P)}")
        return P.state == solved_state

    print(f"    --- {appclass}: solve it ('q' to give up) ---")
    while P.state != solved_state:
        shown = {k: v for k, v in vars(P).items() if not k.startswith("_")
                 and k not in ("state", "initial_state", "state_constants")}
        print(f"    [{P.state}] {shown}")
        opts = [(n, t) for n, _, t in sub._enabled()]
        for i, (_, t) in enumerate(opts):
            print(f"      {i:2}. {sub_label(t)}")
        raw = input("    puzzle > ").strip()
        if raw in ("q", "quit"):
            return False
        if not raw.isdigit() or int(raw) >= len(opts):
            print("      ?")
            continue
        try:
            sub.fire(opts[int(raw)][0])
        except App.InvariantViolation:
            break
    print("    *** the puzzle yields. ***")
    return True


def run_host(app, alias: str, interactive: bool = True) -> bool:
    """A stub is Located (the player attempted it). Enforce the clue
    precondition, run the puzzle, then fire Solved so the age unlocks."""
    if app._components[CLUE_OF[alias]].state != "Seen":
        # The menu gates on this, so we should not reach here in normal play.
        print("  You fiddle with it, but have no idea how -- find the clue first.")
        return False
    if alias in SUBGAME:
        bindir, appclass = SUBGAME[alias]
        _say(f"the host runs the {appclass} sub-game (its own verified model):")
        ok = run_subgame(bindir, appclass, SOLUTION[appclass] if not interactive else None)
        if not ok:
            print("  (you step away -- the puzzle is unsolved)")
            return False
    else:  # DunnyPuz: the fireplace -- enter the pattern, no sub-model
        if interactive:
            input("  Enter the fireplace marker pattern (from the completed book) > ")
        print("  The marker pattern matches; the fireplace wall turns.")
    getattr(app.p, resolve_attr(alias))("Solved")
    print(f"  >>> the host reports {alias} solved -- the way opens.")
    return True


# --------------------------------------------------------------------------- #
# the player loop
# --------------------------------------------------------------------------- #

def status(app) -> str:
    """Player status -- like play_nav's, but keyed off this model's real clue
    component names (the Dunny clue is FireplaceCombo, not ClueDunny)."""
    c = app._components
    here = c["PL"].state
    clues = [age for age, stub in AGE2STUB.items() if c[CLUE_OF[stub]].state == "Seen"]
    solved = [age for age, stub in AGE2STUB.items() if c[stub].state == "Solved"]
    see = ""
    for age, card in N.CLUE_CARDS.items():
        if here == card:
            see = f"\n  You see the {age} clue here. (read it)"
    pages = [n for n in N.PAGE_NAMES if ("Page" + n) in c]   # this build's colour
    carrying = next((n for n in pages if c["Page" + n].state == "Held"), None)
    delivered = sum(1 for n in pages if c["Page" + n].state == "Gone")
    hands = "carrying " + carrying if carrying else "empty-handed"
    return (f"\n== You are at {N.pretty(here)} =={see}\n"
            f"  clues read: {', '.join(clues) or '(none)'}"
            f"   |   ages solved: {', '.join(solved) or '(none)'}\n"
            f"  hands: {hands}   |   pages delivered: {delivered}/8")


def menu(app):
    """Player choices, with attempts hidden until their clue is in hand."""
    out = []
    for n, label, t in N.moves(app):
        age = attempt_age(t)
        if age and app._components[CLUE_OF[AGE2STUB[age]]].state != "Seen":
            continue                      # can't attempt without the clue yet
        out.append((n, label, t))
    return out


def do_imager(app, name, t) -> bool:
    """Intercept the imager transition: ask for the count, validate it against
    the live switches-On total, fire only if right."""
    need = switches_on(app)
    raw = input(f"  The imager waits for a number. How many marker switches did "
                f"you turn on? > ").strip()
    if not raw.isdigit() or int(raw) != need:
        print(f"  The imager rejects {raw or '(nothing)'} and resets.")
        return False
    app.fire(name)
    print(f"  You enter {need}. Atrus's message plays: the tower rotation is revealed.")
    return True


def main() -> int:
    app = N.NavMyst()
    # nav-myst carries the campaign's staging canaries (Seg0_CluesNotRead, ...) --
    # "you cannot reach this milestone" probes. In a live PLAYthrough reaching them
    # is the point, so swallow them (they fire AFTER the transition commits).
    app._milestones = []
    app.on_invariant_violated = lambda exc: app._milestones.append(exc.inv_id)
    mode = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "play"
    if mode == "demo":
        return demo(app)
    if mode == "autowin":
        return autowin(app)

    print(__doc__.splitlines()[0])
    while True:
        settle(app)
        for alias in CLUE_OF:                         # any attempted stub waiting?
            if app._components[alias].state == "Located":
                run_host(app, alias, interactive=True)
                settle(app)
        opts = menu(app)
        if not opts:
            print("\nNo moves left -- stuck.")
            return 1
        print(status(app))
        for i, (_, label, _t) in enumerate(opts):
            print(f"  {i:2}. {label}")
        raw = input("move (number / q): ").strip()
        if raw in ("q", "quit"):
            return 0
        if not raw.isdigit() or int(raw) >= len(opts):
            print("  ?")
            continue
        name, _, t = opts[int(raw)]
        if is_imager(t):
            do_imager(app, name, t)
        elif t.to_state in INTERIOR_GATES:
            if pass_gate(app, t.to_state, interactive=True):
                app.fire(name)
        else:
            app.fire(name)


# --------------------------------------------------------------------------- #
# smoke test: prove the host can run every puzzle, and the resolve loop unlocks
# --------------------------------------------------------------------------- #

def demo(app) -> int:
    rc = 0
    print("== host self-test: every access puzzle, run as its own sub-game ==")
    for alias, (bindir, appclass) in SUBGAME.items():
        ok = run_subgame(bindir, appclass, SOLUTION[appclass])
        print(f"  {appclass:16} ({alias:14}) -> {'SOLVED' if ok else 'FAILED'}")
        rc |= 0 if ok else 1

    print("\n== integration: attempt -> host -> resolve -> the age unlocks ==")
    adj = N._walk_adj()
    here = app._components["PL"].state
    path = N._bfs(adj, here, "GearGate2C840")
    if not path:
        print("  !! no walk path to the Mechanical gate")
        return 1
    for nxt in path[1:]:
        assert N._fire_to(app, nxt), f"could not step to {nxt}"
        settle(app)
    assert N._fire_action(app, "solve the Mechanical access"), "attempt unavailable"
    settle(app)
    located = app._components["MechanicalPuz"].state
    print(f"  attempted at the gate -> MechanicalPuz = {located}")
    assert located == "Located", "stub did not fire to the host"
    # the host runs the clock-tower sub-game and reports Solved. (We drive the
    # clue precondition directly here -- the prologue that reveals it is its own
    # 150-step segment; this smoke isolates the host loop.)
    app._components["ClueMechanical"].state = "Seen"
    assert run_host(app, "MechanicalPuz", interactive=False), "host failed to solve"
    solved = app._components["MechanicalPuz"].state
    print(f"  host fired Solved -> MechanicalPuz = {solved}")
    assert solved == "Solved", "resolve did not land"
    settle(app)
    takeable = any("mechanical linking book" in lbl.lower() for _, lbl, _ in N.moves(app))
    print(f"  'take the Mechanical linking book' now offered: {takeable}")
    assert takeable, "age did not unlock after solve"

    print("\n== imager: the prologue 'enter the number' host (validates the count) ==")
    app._components["NoteClue"].state = "Seen"
    for s in SWITCHES:
        app._components[f"Switch{s}"].state = "On"
    path = N._bfs(N._walk_adj(), app._components["PL"].state, "VaultcontrolpanelC948")
    assert path, "no walk path to the imager"
    for nxt in path[1:]:
        assert N._fire_to(app, nxt), f"could not step to {nxt}"
        settle(app)
    iv = [(n, t) for n, _, t in N.moves(app) if is_imager(t)]
    assert iv, "imager move not offered with note + 8 switches on"
    name, t = iv[0]
    import io
    sys.stdin = io.StringIO("7\n")                    # wrong count -> rejected
    assert do_imager(app, name, t) is False, "imager accepted a wrong count"
    sys.stdin = io.StringIO("8\n")                    # right count -> fires
    assert do_imager(app, name, t) is True, "imager rejected the right count"
    settle(app)
    revealed = app._components["TowerRevealed"].state
    print(f"  entered 8 (=switches on) -> TowerRevealed = {revealed}")
    assert revealed == "Seen", "tower not revealed after the imager"

    print(f"\n{'ALL HOST CHECKS PASS' if rc == 0 else 'SOME PUZZLES FAILED'}")
    return rc


# --------------------------------------------------------------------------- #
# autowin: drive the WHOLE game to a freed Atrus through the host harness
# --------------------------------------------------------------------------- #
#
# This is the end-to-end proof: not a replay of a pre-baked trace, but a live
# playthrough of the unified nav-myst model. We supply the high-level plan (the
# causal order a player follows); navigation between objective cards is BFS over
# the model's own movement graph, every access puzzle is SOLVED by the host
# running its real sub-game, and the imager is the host count. The model's guards
# enforce the rest -- if a step were out of order, its action would not be
# enabled and the run would stop. Success is the model reaching Atrus = Freed.

import json as _json
from collections import deque as _deque


def _move_graph():
    """All player movement edges (walks + the 10 link/return edges) from the
    PlayerLocation source -- the graph BFS navigates."""
    pl = _json.loads((HERE / "lib" / "PlayerLocation.kinner.json").read_text("utf-8"))
    adj: dict = {}
    for t in pl["actors"][0]["triples"]:
        a, b = t["currentState"], t["newState"]
        if a != b:
            adj.setdefault(a, []).append(b)
    return adj


def _bfs(adj, start, goal, blocked):
    seen, q = {start: None}, _deque([start])
    while q:
        x = q.popleft()
        if x == goal:
            path = [x]
            while seen[path[-1]] is not None:
                path.append(seen[path[-1]])
            return path[::-1]
        for y in adj.get(x, []):
            if y not in seen and (x, y) not in blocked:
                seen[y] = x
                q.append(y)
    return None


def pass_gate(app, card, interactive=True) -> bool:
    """A card may be gated by interior host puzzles (Selenitic: the sound puzzle
    opens the maze-hut door, then you ride the maze). Run them once before entering.
    Returns False only if a sub-game is left unsolved (interactive give-up)."""
    for bindir, appclass, solved in INTERIOR_GATES.get(card, []):
        if appclass in _solved_gates:
            continue
        _say(f"--- the host runs {appclass} (its own verified model) ---")
        script = None if interactive else SOLUTION[appclass]
        if not run_subgame(bindir, appclass, script, solved):
            return False
        _solved_gates.add(appclass)
    return True


def goto(app, target, adj) -> None:
    """Walk/link the player to target, BFS over the live model. An edge that is
    not currently enabled (e.g. an age link before its puzzle is solved) gets
    blocked and the route is recomputed -- so goto only ever fires real moves.
    Interior gates (e.g. the Selenitic maze-hut door) run their host sub-games
    before the player may pass."""
    blocked = set()
    for _ in range(4000):
        here = app._components["PL"].state
        if here == target:
            return
        path = _bfs(adj, here, target, blocked)
        if not path or len(path) < 2:
            raise RuntimeError(f"no route {here} -> {target} (blocked {len(blocked)})")
        nxt = path[1]
        if nxt in INTERIOR_GATES and not pass_gate(app, nxt, interactive=False):
            blocked.add((here, nxt))
            continue
        if N._fire_to(app, nxt):
            ev = N.EVENT.get((here, nxt), "")
            if ev.startswith("walks to"):
                _log(f"walk -> {N.pretty(nxt)}", "walk")
            else:                                  # a linking book between ages
                _log(f"link -> {N.pretty(nxt)}  ({ev})", "book")
            settle(app)
        else:
            blocked.add((here, nxt))               # edge not enabled now
    raise RuntimeError(f"goto {target} did not converge")


def act(app, needle: str) -> None:
    nd = N._norm(needle)
    for n, label, _t in N.moves(app):
        if nd in N._norm(label):
            _log(f"** {label} **", _classify(label))
            app.fire(n)
            settle(app)
            return
    avail = [lbl for _, lbl, _ in N.moves(app)]
    hands = app._components["Hands"].state
    raise RuntimeError(f"action {needle!r} unavailable at {app._components['PL'].state} "
                       f"(Hands={hands}); available: {avail}")


SWITCH_CARDS = [("Dock", "Dock2EC689"), ("Cabin", "Cabin1EC726"),
                ("Clock", "Clock1EC869"), ("Gears", "Gears1NEC628"),
                ("SpaceShip", "SpaceShipPath1EC634"), ("Generator", "GenStairs1EC713"),
                ("Tree", "TreeGate3EC707"), ("Planetarium", "Planetarium1SLightsOnC699")]
CLUE_STEPS = [("Mechanical", "ObservCodeGearC864"), ("Stoneship", "ObservCodeBoatC847"),
              ("Channelwood", "ObservCodeTreeC867"), ("Selenitic", "ObservCodeSpaceC866")]
# age: (name, gate card, stub, in-age page card)
AGE_STEPS = [("Mechanical", "GearGate2C840", "MechanicalPuz", "SCacheChestC455"),
             ("Stoneship", "Boat5WopenC891", "StoneshipPuz", "SirrusDresser2C1320"),
             ("Channelwood", "TreeRoom3NopenC894", "ChannelwoodPuz", "SirrusNorthDesknC362"),
             ("Selenitic", "SpaceShip2NC805", "SeleniticPuz", "StoneForestMicStandC990")]


def autowin(app) -> int:
    global _VERBOSE
    _VERBOSE = True
    _STEP[0] = 0
    _solved_gates.clear()
    adj = _move_graph()

    print("Legend:  " + _color("walk", "walk") + "   "
          + _color("picked up a page", "item") + "   "
          + _color("read a clue", "clue") + "   "
          + _color("linked between ages (book)", "book"))

    def banner(s):
        print(f"\n--- {s} ---")

    banner("Prologue: Catherine's note")
    goto(app, "Woodpath1NnoteC684", adj)
    act(app, "read Catherine")

    banner("Prologue: flip all eight marker switches")
    for name, card in SWITCH_CARDS:
        goto(app, card, adj)
        act(app, f"flip the {name} marker switch")
    print(f"  switches on: {switches_on(app)}/8")

    banner("Prologue: the imager (host enters the count)")
    goto(app, "VaultcontrolpanelC948", adj)
    iv = [(n, t) for n, _, t in N.moves(app) if is_imager(t)]
    assert iv, "imager not available"
    _log(f"** enter the marker count ({switches_on(app)}) at the imager **")
    app.fire(iv[0][0])                          # count is 8 by construction
    settle(app)
    print(f"          -> TowerRevealed = {app._components['TowerRevealed'].state}")

    banner("Read the four tower clues")
    for age, card in CLUE_STEPS:
        goto(app, card, adj)
        act(app, f"read the {age} clue")

    banner("Read the four journals on the library bookshelf (the solution clues)")
    goto(app, "Library1NC773", adj)
    for age, _card in CLUE_STEPS:
        act(app, f"read the {age} journal")
    act(app, "read the Pattern Book")            # the fifth shelf book (fireplace pattern)

    for age, gate, stub, pagecard in AGE_STEPS:
        banner(f"{age}: attempt -> HOST runs the sub-game -> grab the page -> deposit")
        goto(app, gate, adj)
        act(app, f"solve the {age} access")          # stub -> Located
        assert run_host(app, stub, interactive=False), f"host failed on {age}"
        settle(app)
        goto(app, pagecard, adj)                     # links into the age
        act(app, f"red page ({age})")
        goto(app, "Library2NC783", adj)              # returns to the island
        act(app, f"place the {age} page")

    banner("The freebie page -> Sirrus's book is complete -> the combo is revealed")
    goto(app, "Library2NC783", adj)
    act(app, "pick up the freebie")
    act(app, "place the freebie")
    print(f"  Sirrus book = {app._components['Sirrus'].state}")

    banner("The fireplace: learn the combination, attempt -> HOST")
    goto(app, "FirePlace1SWDoorUpC871", adj)
    act(app, "learn the fireplace combination")
    act(app, "solve the Dunny access")               # stub -> Located
    assert run_host(app, "DunnyPuz", interactive=False), "host failed on the fireplace"
    settle(app)

    banner("Carry both white-page halves through D'ni into Atrus's green book")
    for age, white_card, restore_needle in [
            ("Stoneship", "Boat1EC1253", "restore the Stoneship white-page half"),
            ("Channelwood", "Boardwalk36SC87", "restore the Channelwood white-page half")]:
        goto(app, white_card, adj)                   # re-enter the age (puzzle still Solved)
        act(app, f"{age} white-page half")
        goto(app, "RockFallOutC398", adj)            # via the fireplace into D'ni
        act(app, restore_needle)
        print(f"  Atrus book = {app._components['Atrus'].state}")

    won = app._components["Atrus"].state == "Freed"
    seen = list(dict.fromkeys(app._milestones))      # distinct, in order
    print(f"\n  staging milestones realized en route ({len(seen)}): {', '.join(seen)}")
    print(f"\n{'*** ATRUS IS FREED -- the whole game, won live through the host. ***' if won else 'DID NOT REACH Atrus=Freed'}")
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
