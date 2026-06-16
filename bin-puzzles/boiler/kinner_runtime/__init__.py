"""
kinner_runtime: minimal runtime for the hand-designed Python target.

Implements the contract established across:
  - design/python-target/invariants.md
  - design/python-target/asymmetries.md
  - design/python-target/transitions.md

Design goal: every method reads top-to-bottom as a straight sequence.
No closures over mutable state, no clever tricks, no hidden dispatch.
This file gets scrutinized for TLC-vs-Python fidelity; it earns that
scrutiny by being obvious at a glance.

Claim: if every transition fires atomically, guards read current
state directly, and hooks run post-commit, then the transition
relation the simulator produces matches TLA+'s. TLC's safety and
liveness proofs transfer to this runtime's runs.
"""
from __future__ import annotations

import keyword
import random
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------


class DisabledActionError(Exception):
    """Raised when `fire()` is called with a disabled action. Carries
    the action name, component state, and reason so a trace script
    sees exactly where it diverged from the spec."""

    def __init__(self, qualified_name: str, component_state: str, reason: str):
        super().__init__(
            f"action {qualified_name!r} not enabled "
            f"(current state {component_state!r}): {reason}"
        )
        self.qualified_name = qualified_name
        self.component_state = component_state
        self.reason = reason


class InvariantViolation(Exception):
    """Raised when a registered invariant fails evaluation. Carries
    the invariant id, a readable state summary, and the action name
    that led to the violating state."""

    def __init__(self, inv_id: str, state_summary: str, last_action: str | None):
        super().__init__(
            f"invariant {inv_id!r} violated after "
            f"{last_action or 'init'}: {state_summary}"
        )
        self.inv_id = inv_id
        self.state_summary = state_summary
        self.last_action = last_action


class MessageSetViolation(Exception):
    """Raised when a send or raise applies a tag outside the
    sender's declared MessageSet (class attr `_message_set`
    populated from the manifest's `tag_constants`).

    TLA+ rejects the same state via TypeInvariant / `\\in MessageSet`
    guards; Python raises at apply time so the offending transition
    is visible at the call site. Untyped components (empty
    `_message_set`) never raise.
    """

    def __init__(self, component_alias: str, port: str, tag: str,
                 allowed: frozenset):
        super().__init__(
            f"component {component_alias!r} sent tag {tag!r} on "
            f"port {port!r} which is not in its declared MessageSet "
            f"{sorted(allowed)!r}"
        )
        self.component_alias = component_alias
        self.port = port
        self.tag = tag
        self.allowed = allowed


class BoundViolation(Exception):
    """Raised when an assign writes an integer outside the variable's declared
    bound (class attr `_typed_var_bounds`, populated from the typed-variable
    declarations).

    TLA+ rejects the same state via the TypeInvariant `v \\in lo..hi`; Python
    raises at apply time so the offending transition is visible at the call site
    (backlog 257 Finding B). Variables without a declared integer bound never
    raise.
    """

    def __init__(self, component_alias: str, name: str, value: Any,
                 lo: int, hi: int):
        super().__init__(
            f"component {component_alias!r} assigned {name!r} = {value} "
            f"outside its declared bound {lo}..{hi}"
        )
        self.component_alias = component_alias
        self.name = name
        self.value = value
        self.lo = lo
        self.hi = hi


class ReplayDivergence(Exception):
    """Raised by `Application.replay(trace)` when Python's post-state
    differs from the trace's recorded post-state. Identifies the
    exact step and the per-variable diffs."""

    def __init__(self, step: int, action: str | None, diffs: dict[str, tuple]):
        diff_summary = ", ".join(
            f"{k}: expected {exp!r}, got {act!r}"
            for k, (exp, act) in diffs.items()
        )
        super().__init__(
            f"replay diverged at step {step} "
            f"({action or 'init'}): {diff_summary}"
        )
        self.step = step
        self.action = action
        self.diffs = diffs


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SendSlot:
    """One outgoing channel effect within a transition. A transition
    fires zero-or-more SendSlots atomically with its receive (if any).
    Backlog 180."""

    port: str
    tag: str | None = None


@dataclass
class Transition:
    """Static description of a component transition. Emitted by the
    code generator inside each Component's `_build_transitions()`.

    A transition fires atomically: optional receive (single channel),
    any number of sends (`sends` tuple), any number of typed-variable
    assigns, optional counter increment, optional effect_fn. Backlog
    076 introduced the multi-effect shape; backlog 180 generalized
    sends from the original two-slot (primary + raise) form to an
    arbitrary-length tuple mirroring the IR.
    """

    name: str
    from_state: str
    to_state: str
    kind: str           # "local" | "send" | "receive" | "send_receive"
    state_field: str = "state"
    """Which state attribute this transition governs. Defaults to the single
    `self.state` FSM every leaf / single-inline composite uses (so existing
    components are byte-identical). A MULTI-inline composite (backlog 242,
    option B: one composite module folds N inline-actor FSMs) emits one
    Transition per actor carrying that actor's own field (`tickState`), so the
    composite Component carries several independent state machines."""
    recv_port: str | None = None
    recv_tag: str | None = None
    sends: tuple[SendSlot, ...] = ()
    guard_fn: Callable[[Any], bool] | None = None
    effect_fn: Callable[[Any], None] | None = None
    assigns: tuple[tuple[str, Callable[[Any], Any]], ...] = ()
    """Typed-variable assignments fired atomically with the
    transition. Each entry is `(var_name, value_fn)` where `value_fn`
    takes the component instance and returns the new value. Simultaneous
    (TLA-prime) semantics: every `value_fn` is evaluated against the
    PRE-state -- before `_apply` writes state_field or channels -- and
    only then are the results written. So no assign observes another's
    write (or any other write this transition makes), and the order is
    immaterial. Backlog 246."""


@dataclass
class Channel:
    """Runtime state of a tagged channel binding. One instance per
    out->in port pair; both endpoints mutate this object directly.
    State progression: NotSent -> InFlight -> Delivered.

    `state_constants` records which family of TLA state constants the
    compiler chose for this channel: `"shared"` means `Ch_NotSent` /
    `Ch_InFlight` / `Ch_Delivered` (cross-component channels), and
    `"per_channel"` means `<VarPascal>_NotSent` / ... (pure-actor
    channels). Default `"shared"` keeps existing call sites correct
    for cross-component channels. Read by the `to_tla_view` projection
    (backlog 201.3) so the TLA-view value encoding matches what the
    `.tla` emit produced, without re-inferring from the runtime side."""

    name: str = ""
    state: str = "NotSent"
    tag: str | None = None
    state_constants: str = "shared"

    def snapshot(self) -> tuple[str, str | None]:
        return (self.state, self.tag)

    def restore(self, snap: tuple[str, str | None]) -> None:
        self.state, self.tag = snap


@dataclass(frozen=True)
class TransitionEvent:
    """Hook payload. Carries everything a hook needs to react --
    action name, component alias, from/to state, optional tag."""

    qualified_name: str
    action_name: str
    component_alias: str
    from_state: str
    to_state: str
    tag: str | None = None
    sends: tuple[tuple[str, str | None], ...] = ()
    """Every (port, tag) this transition emitted, in SendSlot order, with
    forward-send tags resolved -- bound and open ports alike (backlog 249
    Part C). The legacy `tag` field keeps its lossy first-slot/recv rule
    for compatibility; this is the faithful payload a host reads to know
    "the component emitted Filled on EVT_OUT"."""


@dataclass
class HostEvent:
    """One component->host trigger firing: an emission on an open out-port,
    appended to the Application's event queue in commit order (backlog 252).
    The queue is the host-side mirror of channel state -- the model's
    "queue" is channel/port state consumed when a receive fires; the host's
    is this, drained at settle. It is host OBSERVATION state, not model
    state: excluded from snapshot()/restore() (you can't unobserve).

    `drained` and `dispatched` are independent consumers' marks: `drained`
    means a `drain()` call returned this event's tag; `dispatched` means
    settle pushed it through the host's trigger handlers. Draining never
    suppresses dispatch, and vice versa (the 249 C independence, kept)."""

    seq: int
    key: tuple[str, str]      # (leaf component alias, leaf port) -- IR names
    tag: str | None
    drained: bool = False
    dispatched: bool = False


@dataclass(frozen=True)
class PumpReport:
    """What one `pump()` -- the world's turn -- committed. `oscillation`
    is the visited-state stop's finding: the cascade revisited a world
    state, so the pump terminated early AND the revisit is reportable
    ("the world oscillates here"), not just tolerated."""

    events: tuple[TransitionEvent, ...]
    oscillation: str | None = None


@dataclass(frozen=True)
class SettleVerdict:
    """What one `settle()` did, returned (and passed to the `on_settle`
    hook). Deadlock becomes a returnable verdict, not a hung process: the
    single-threaded loop KNOWS, deterministically, when both queues are
    empty and nothing relevant is enabled."""

    steps: tuple[TransitionEvent, ...]
    """World moves committed during this settle, commit order."""
    dispatched: tuple[HostEvent, ...]
    """Host events drained through trigger handlers, queue order."""
    rounds: int
    """Dispatch rounds run (each: pump to quiet, then drain one batch)."""
    oscillation: str | None
    """The pump's visited-state finding, if the world cycled."""
    livelock: bool
    """True when a (state, queue) configuration repeated -- handler
    ping-pong, Node's self-requeueing-microtask starvation verbatim."""
    enabled: tuple[str, ...]
    """The background: every action still enabled when settle returned."""
    deferred: bool = False
    """True when settle() was called from inside a running settle (or
    mid-commit, from a hook): the call is a no-op and the OUTER loop owns
    the cascade -- handler gestures are next-round tasks, never nested."""

    @property
    def quiescent(self) -> bool:
        """Clean exit: the queue drained and the world's wake exhausted,
        with no oscillation or livelock finding."""
        return (not self.deferred and not self.livelock
                and self.oscillation is None)


# ---------------------------------------------------------------------------
# Synthesized actions (fan-out, fan-in)
#
# Each subclass carries exactly the data it needs to check `is_enabled`
# and perform `apply`. No closures, no late-binding traps. A reviewer
# can read each class in under 15 lines and know what fires.
# ---------------------------------------------------------------------------


class SynthAction:
    """Application-level action synthesized at bind time. Fires like
    a transition but belongs to no component."""

    name: str

    def is_enabled(self) -> bool:
        raise NotImplementedError

    def apply(self) -> None:
        raise NotImplementedError


class FanOut(SynthAction):
    """Broadcast one sender's InFlight message to N receiver channels
    atomically. Source becomes Delivered; each target becomes InFlight
    with the same tag."""

    def __init__(self, name: str, source: Channel, targets: list[Channel]):
        self.name = name
        self.source = source
        self.targets = targets

    def is_enabled(self) -> bool:
        if self.source.state != "InFlight":
            return False
        return all(t.state in ("NotSent", "Delivered") for t in self.targets)

    def apply(self) -> None:
        tag = self.source.tag
        for t in self.targets:
            t.tag = tag
            t.state = "InFlight"
        self.source.state = "Delivered"


class FanInMerge(SynthAction):
    """Merge one source into the shared destination. Source becomes
    Delivered; destination becomes InFlight with the source's tag.
    One such action exists per sender in a fan-in group."""

    def __init__(self, name: str, source: Channel, dest: Channel):
        self.name = name
        self.source = source
        self.dest = dest

    def is_enabled(self) -> bool:
        return (
            self.source.state == "InFlight"
            and self.dest.state in ("NotSent", "Delivered")
        )

    def apply(self) -> None:
        self.dest.tag = self.source.tag
        self.dest.state = "InFlight"
        self.source.state = "Delivered"


# ---------------------------------------------------------------------------
# Buffered ports (backlog 077)
#
# A composite MODULE can declare `bufferDepth` on an external port.
# The port's external surface is unchanged -- it still appears in the
# composite's `_in_ports` or `_out_ports` and callers bind it via
# `_channel` like any other port. What differs is internal: instead
# of forwarding to the sub-component's port, the composite interposes
# a bounded Seq buffer between the external channel and the sub-
# component's channel. Two synthesized actions per buffer move
# messages: enqueue (source channel -> buffer) and dequeue (buffer ->
# target channel). Depth is a per-buffer cap; enqueue is disabled
# when `len(items) >= depth`, matching TLA+'s `Len(buf) < bound`.
# ---------------------------------------------------------------------------


@dataclass
class Buffer:
    """FIFO message buffer for a composite's buffered port. Owned by
    the composite instance; snapshotted by the Application. A buffer
    has fixed `depth` (set in the composite's __init__) and a list of
    tags (`items`). Enqueue appends to the tail; dequeue pops from
    the head. Class-attribute membership of the associated port in
    `_in_ports`/`_out_ports` is unchanged -- only internal wiring
    differs between a buffered and a non-buffered port."""

    name: str = ""
    depth: int = 0
    items: list = field(default_factory=list)

    def snapshot(self) -> tuple[int, tuple]:
        return (self.depth, tuple(self.items))

    def restore(self, snap: tuple[int, tuple]) -> None:
        depth, items = snap
        self.depth = depth
        self.items = list(items)


class BufferEnqueue(SynthAction):
    """Enqueue action: when the source channel is InFlight AND the
    buffer has room, append the source's tag to the buffer and mark
    the source Delivered. Matches TLA+'s synthesized Enqueue action
    body exactly (see PeekLockQueue.tla backlog 077).

    Source is resolved lazily via `(component, port)` lookup on
    `_port_channels` at fire time. This lets external callers bind
    the composite's port any time before firing -- init order is
    irrelevant."""

    def __init__(
        self,
        name: str,
        source_comp: "Component",
        source_port: str,
        buffer: Buffer,
    ):
        self.name = name
        self.source_comp = source_comp
        self.source_port = source_port
        self.buffer = buffer

    def _source(self) -> Channel | None:
        return self.source_comp._port_channels.get(self.source_port)

    def is_enabled(self) -> bool:
        src = self._source()
        if src is None:
            return False
        return (
            src.state == "InFlight"
            and len(self.buffer.items) < self.buffer.depth
        )

    def apply(self) -> None:
        src = self._source()
        assert src is not None, (
            f"{self.name}: source channel disappeared between "
            f"is_enabled and apply"
        )
        # Backlog 201.7: untagged in-flight enqueues store the
        # explicit 'Ch_InFlight' sentinel so the buffer's contents
        # match TLA+'s Append(buffer, chan) when `chan = Ch_InFlight`.
        # BufferDequeue translates back to `tag=None` on dequeue.
        self.buffer.items.append(
            src.tag if src.tag is not None else "Ch_InFlight"
        )
        src.state = "Delivered"


class BufferDequeue(SynthAction):
    """Dequeue action: when the buffer is non-empty AND the target
    channel is clear (NotSent or Delivered), pop the head tag into
    the target as InFlight.

    Target is resolved lazily via `(component, port)` lookup on
    `_port_channels` at fire time."""

    def __init__(
        self,
        name: str,
        buffer: Buffer,
        target_comp: "Component",
        target_port: str,
    ):
        self.name = name
        self.buffer = buffer
        self.target_comp = target_comp
        self.target_port = target_port

    def _target(self) -> Channel | None:
        return self.target_comp._port_channels.get(self.target_port)

    def is_enabled(self) -> bool:
        tgt = self._target()
        if tgt is None:
            return False
        return (
            len(self.buffer.items) > 0
            and tgt.state in ("NotSent", "Delivered")
        )

    def apply(self) -> None:
        tgt = self._target()
        assert tgt is not None, (
            f"{self.name}: target channel disappeared between "
            f"is_enabled and apply"
        )
        # Backlog 201.7: BufferEnqueue stores 'Ch_InFlight' for
        # untagged in-flight captures so the buffer contents match
        # TLA+. Translate back to `tag=None` on dequeue so the
        # downstream receiver sees the normal untagged-in-flight
        # encoding (state=InFlight, tag=None).
        item = self.buffer.items.pop(0)
        tgt.tag = None if item == "Ch_InFlight" else item
        tgt.state = "InFlight"


class FanOutBufferAppend(SynthAction):
    """Atomic N-way buffered enqueue (backlog 074.1 + 103). When the
    source channel is InFlight AND every target buffer has room,
    append the source's tag to each buffer and mark the source
    Delivered. Matches TLA+'s `FanOutAppend` action exactly:
    one publisher, N per-subscriber Seq buffers, one atomic step.

    The per-subscriber `DrainAction` (rendered as `BufferDequeue` in
    Python) fires independently and pops each buffer into its
    subscriber's input channel. Source resolved lazily via
    `(component, port)` lookup so binding order is irrelevant.
    """

    def __init__(
        self,
        name: str,
        source_comp: "Component",
        source_port: str,
        buffers: list[Buffer],
    ):
        self.name = name
        self.source_comp = source_comp
        self.source_port = source_port
        self.buffers = buffers

    def _source(self) -> Channel | None:
        return self.source_comp._port_channels.get(self.source_port)

    def is_enabled(self) -> bool:
        src = self._source()
        if src is None:
            return False
        if src.state != "InFlight":
            return False
        return all(len(b.items) < b.depth for b in self.buffers)

    def apply(self) -> None:
        src = self._source()
        assert src is not None, (
            f"{self.name}: source channel disappeared between "
            f"is_enabled and apply"
        )
        for b in self.buffers:
            b.items.append(src.tag)
        src.state = "Delivered"


# ---------------------------------------------------------------------------
# Component base class
# ---------------------------------------------------------------------------


class Component:
    """Base class for every generated component. Subclasses declare:

      - `initial_state`: str
      - `state_constants`: tuple of state names
      - `_in_ports` / `_out_ports` / `_observe_ports`: tuples
      - `_typed_var_defaults`: {name: initial_value}

    and implement `_build_transitions() -> list[Transition]`.
    """

    initial_state: str = ""
    state_constants: tuple[str, ...] = ()
    _in_ports: tuple[str, ...] = ()
    _out_ports: tuple[str, ...] = ()
    _observe_ports: tuple[str, ...] = ()
    # Option defaults from the manifest. Instance __init__ seeds
    # self._options = dict(_option_defaults); per-instance overrides
    # merge via **kwargs at construction time. Guards translate
    # CONSTANT references (the option name) to self._options[name]
    # so bound-checking transitions like `count < maxWrites` resolve
    # at runtime.
    # Backlog 217 C2: the class-level defaults are immutable
    # (`MappingProxyType` / `frozenset`) so a subclass that mutates one
    # in place -- instead of redeclaring -- raises instead of silently
    # corrupting the base dict every other component inherits. Generated
    # subclasses always redeclare these with plain dict literals, which
    # is unaffected; `__init__` only ever reads + copies them.
    _option_defaults: Mapping[str, Any] = MappingProxyType({})
    # CONSTANT name -> option name reverse map. Parent composites
    # and APP codegen pass option overrides via the CONSTANT name
    # found in `ComponentInstance.with_mappings` (the same path TLA+
    # uses for its INSTANCE ... WITH clause). __init__ translates
    # each CONSTANT-name kwarg to its option-name key via this dict.
    # Emitted by leaf codegen from `manifest.options[name].constant`.
    _option_constants: Mapping[str, str] = MappingProxyType({})
    # Backlog 134: typed actor variables -- name -> default value.
    # __init__ seeds each as an instance attribute; transitions with
    # `assigns` mutate them via `setattr(comp, name, value_fn(comp))`.
    _typed_var_defaults: Mapping[str, Any] = MappingProxyType({})
    # Backlog 257: integer typed-var bounds -- name -> (lo, hi). Mirrors the TLA
    # TypeInvariant `v \\in lo..hi`; an assign writing outside the bound raises
    # BoundViolation at apply time. Empty for bool/unbounded vars.
    _typed_var_bounds: Mapping[str, Any] = MappingProxyType({})
    # Allowed tag values for sends + raises. Emitted by leaf
    # codegen from `manifest.tag_constants` -- the same IR field
    # TLA+ reads to emit TypeInvariant `chan \\in ... \\union
    # MessageSet`. Empty set means untyped: the component predates
    # tagConstants or opts out, and any tag (including None) is
    # accepted. Runtime _apply enforces on every SendSlot's non-None
    # tag against a non-empty set. See MessageSetViolation.
    _message_set: frozenset = frozenset()
    # Backlog 242: extra named state machines a MULTI-inline composite carries
    # (option B -- one composite module folds N inline-actor FSMs, mirroring the
    # TLA composite that declares N state vars). `{field: (initial, (states...))}`.
    # Empty for every single-FSM component (every leaf, every single-inline
    # composite), which keeps using only `self.state` -- so the whole multi-FSM
    # path is opt-in and default-safe. Each transition names which field it
    # governs via `Transition.state_field`.
    _state_fields: Mapping[str, tuple] = MappingProxyType({})

    def __init__(self, **kwargs: Any) -> None:
        self.state: str = self.initial_state
        self._options: dict[str, Any] = dict(self._option_defaults)
        # Backlog 148: merge kwargs into self._options BEFORE seeding
        # typed-var attrs so option-name defaults can read the resolved
        # option value (post-override). Pre-148 ordering seeded typed
        # vars first; harmless when defaults were always int literals.
        for k, v in kwargs.items():
            # Kwargs may arrive keyed by option name (direct user
            # overrides) or CONSTANT name (parent composite / APP
            # codegen mirroring TLA+'s WITH clause).
            if k in self._option_defaults:
                option_name = k
            elif k in self._option_constants:
                option_name = self._option_constants[k]
            else:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected kwarg "
                    f"{k!r}; known options: {list(self._option_defaults)}; "
                    f"known CONSTANT aliases: {list(self._option_constants)}"
                )
            self._options[option_name] = v
        # Seed typed-var attrs. int default => set directly; str default
        # (backlog 148) names an option, look up the resolved value.
        for _tv_name, _tv_default in self._typed_var_defaults.items():
            if isinstance(_tv_default, str):
                setattr(self, _tv_name, self._options[_tv_default])
            else:
                setattr(self, _tv_name, _tv_default)
        # Backlog 242: seed each extra FSM state (a multi-inline composite).
        # No-op for single-FSM components (`_state_fields` empty).
        for _sf_name, (_sf_init, _sf_consts) in self._state_fields.items():
            setattr(self, _sf_name, _sf_init)
        self._port_channels: dict[str, Channel] = {}
        self._observed: dict[str, Component] = {}
        self._transitions: list[Transition] = self._build_transitions()
        self._app: Application | None = None
        self._alias: str = ""

    def _build_transitions(self) -> list[Transition]:
        raise NotImplementedError

    def patch(self, **updates: Any) -> None:
        """Hot-patch state and/or typed variables. Explicit escape
        hatch from the verified transition relation. Validates names
        so typos fail loud."""
        if "state" in updates:
            new_state = updates.pop("state")
            if new_state not in self.state_constants:
                raise ValueError(
                    f"patch: {new_state!r} not in {self._alias}'s "
                    f"state_constants {self.state_constants}"
                )
            self.state = new_state
        for key, val in updates.items():
            if key not in self._typed_var_defaults:
                raise ValueError(
                    f"patch: unknown field {key!r} on {self._alias} "
                    f"(typed vars: {list(self._typed_var_defaults)})"
                )
            setattr(self, key, val)


# ---------------------------------------------------------------------------
# Host port surface (backlog 249 Part B)
#
# `app.p.<port>(tag)` -- the host-injection point for an OPEN input port.
# Runtime-synthesized: openness is an instance-level fact (`_is_open_input`),
# so the namespace is discovered from the live instance, never emitted.
# Names project from the IR port name carried verbatim in `_in_ports`.
# ---------------------------------------------------------------------------


def _snake(raw: str) -> str:
    """snake_case one IR name segment: camel boundaries split,
    non-alphanumerics (compound-alias dots) become underscores."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).lower().strip("_")


def _surface_name(*segments: str) -> str:
    """Deterministic projection of IR name segment(s) -- a port name,
    optionally alias-qualified -- to a Python attribute. The identifier /
    keyword fix applies ONCE, to the joined result (`IN` -> `in_`, but
    `A` + `IN` -> `a_in`). NOT a flat-name parse -- the inputs are the
    structural names the IR carries."""
    s = "_".join(_snake(seg) for seg in segments)
    if not s or s[0].isdigit():
        s = "_" + s
    if keyword.iskeyword(s):
        s += "_"
    return s


class OpenInPort:
    """Callable host-injection point for one open input port.

    `port_obj(tag)` delivers-and-dispatches in one atomic step: it fires
    the enabled collapsed transition originating from `(port, tag)`. The
    matching rule is exactly the channel rule (`_channel_has_inflight`)
    with the call's tag standing in for the in-flight carrier --

        transition matches iff recv_port == port
                            and (recv_tag is None or recv_tag == tag)

    -- so the receiver cannot tell a host call from a sibling's send
    (the host-model soundness claim; design/python-target/host-model.md).

    Error split mirrors the observe seam: a tag NO transition on this port
    could ever receive (and no wildcard exists, or it is outside a non-empty
    MessageSet) raises ValueError regardless of state; an in-alphabet tag
    with nothing currently enabled raises DisabledActionError. When more
    than one matching transition is enabled -- the model's own don't-care
    branch, which TLA explores -- the app's seeded rng picks, exactly as
    `step()` would have after a sibling's send parked the message. LIVE
    drive only: traces journal the resolved `action_name` and replay drives
    by `fire()`, so no rng ever participates in a replay (249bc-proposal).

    `display` is the outermost (alias, port) the namespace named this entry
    after; firing always targets the resolved LEAF endpoint, where the
    receive transitions live (a composite interface port forwards down).
    """

    def __init__(self, app: "Application", leaf_comp: Component,
                 leaf_port: str, display: str):
        self._app = app
        self._comp = leaf_comp
        self._port = leaf_port
        self._display = display

    def __repr__(self) -> str:
        return f"<OpenInPort {self._display} -> {self._comp._alias}.{self._port}>"

    def __call__(self, tag: str | None = None) -> TransitionEvent:
        app, comp, port = self._app, self._comp, self._port
        app._begin_cascade()    # a host GESTURE opens the cascade (252)
        ts = [t for t in comp._transitions if t.recv_port == port]
        alphabet = {t.recv_tag for t in ts}
        wildcard = None in alphabet
        in_alphabet = (wildcard if tag is None
                       else (tag in alphabet or wildcard)
                       and (not comp._message_set or tag in comp._message_set))
        if not in_alphabet:
            shown = sorted(t for t in alphabet if t is not None)
            raise ValueError(
                f"{self._display}: tag {tag!r} is outside the port's alphabet "
                f"{shown}{' + wildcard' if wildcard else ''}"
                + (f" (MessageSet {sorted(comp._message_set)})"
                   if comp._message_set else "")
            )
        matching = [t for t in ts if t.recv_tag is None or t.recv_tag == tag]
        enabled = [t for t in matching
                   if getattr(comp, t.state_field) == t.from_state
                   and app._transition_enabled(comp, t)]
        if not enabled:
            candidates = ", ".join(
                f"{t.name}[from {t.from_state}]" for t in matching)
            raise DisabledActionError(
                f"{self._display}({tag!r})", getattr(comp, "state", "?"),
                f"no enabled transition for ({port}, {tag!r}); "
                f"candidates: {candidates}",
            )
        t = enabled[0] if len(enabled) == 1 else app._rng.choice(enabled)
        event = app._commit(comp._alias, comp, t)
        app._maybe_auto_settle()    # Node mode folds settle+dispatch in
        return event

    def enabled_tags(self) -> list:
        """The tags this port would accept RIGHT NOW: every recv_tag with at
        least one enabled receive, declaration order, deduped (fork arms --
        same tag, several enabled transitions -- appear once; the call's
        seeded-rng pick resolves which arm fires). A wildcard receive
        (recv_tag None) appears as None.

        This is the public form of the host-menu / witness-bridge query:
        "what commands does the model accept here?" Consumers previously
        read _transitions / _transition_enabled directly, which breaks
        whenever the private surface moves."""
        comp = self._comp
        out: list = []
        for t in comp._transitions:
            if t.recv_port != self._port:
                continue
            if getattr(comp, t.state_field) != t.from_state:
                continue
            if not self._app._transition_enabled(comp, t):
                continue
            if t.recv_tag not in out:
                out.append(t.recv_tag)
        return out


class OpenOutPort:
    """Host-observation point for one open output port (backlog 249 Part C;
    timing redefined by backlog 252).

    Parity is the 182/A-strict argument: an open out never blocks (the
    environment is the most-general receiver), so observing it adds no
    reachable state. Each emission on an open out-port is a component->host
    trigger firing: it enqueues a HostEvent on the Application's event
    queue at apply time. This object is the per-port read surface:

      drain()        -- read MY unprocessed slice of the event queue: tags
                        emitted on this port and not yet drained (FIFO).
                        A pull, available any time; independent of settle.
      subscribe(fn)  -- register fn as a trigger handler: fn(tag) fires off
                        the event queue AT SETTLE, against the settled
                        world (backlog 252 decision 2 -- hooks watch steps;
                        trigger handlers hear settled events). fn may fire
                        gestures: they commit atomically and their
                        consequences are next-round tasks.

    Emissions are host observation history, NOT model state: excluded from
    snapshot()/restore() (you can't unobserve) and unbounded (draining is
    the host's job; tags are small strings)."""

    def __init__(self, app: "Application", leaf_comp: Component,
                 leaf_port: str, display: str):
        self._app = app
        self._comp = leaf_comp
        self._port = leaf_port
        self._display = display

    def __repr__(self) -> str:
        return f"<OpenOutPort {self._display} -> {self._comp._alias}.{self._port}>"

    @property
    def _key(self) -> tuple[str, str]:
        return (self._comp._alias, self._port)

    def drain(self) -> list:
        out = []
        for e in self._app._event_queue:
            if e.key == self._key and not e.drained:
                e.drained = True
                out.append(e.tag)
        return out

    def subscribe(self, fn: Callable[[str | None], None]) -> None:
        self._app._out_subscribers.setdefault(self._key, []).append(fn)


class PortNamespace:
    """The `app.p` attribute container. Plain attributes per open port;
    unknown names fail with the available list."""

    def __init__(self, ports: dict[str, Any]):
        self._ports = dict(ports)
        self.__dict__.update(ports)

    def __getattr__(self, name: str):
        raise AttributeError(
            f"no open port {name!r}; open ports: "
            f"{sorted(self.__dict__.get('_ports', {}))}"
        )

    def __iter__(self):
        return iter(self._ports.items())

    def __repr__(self) -> str:
        return f"<PortNamespace {sorted(self._ports)}>"


# ---------------------------------------------------------------------------
# Application base class
#
# Organized into sections, each a self-contained responsibility:
#   - Registration / binding
#   - Enabled-set enumeration
#   - Fire (explicit) / step (scheduler)
#   - Atomic commit phases (apply -> hooks -> invariants)
#   - Hook dispatch
#   - Snapshot / restore / replay
# ---------------------------------------------------------------------------


class Application:
    """Base class for every generated Application."""

    auto_settle: bool = False
    """Node mode (backlog 252 decision 3): when True, every host gesture
    (`app.p.<port>(tag)` / `fire()`) folds settle+dispatch in -- the world
    settles and the event queue drains through trigger handlers before the
    gesture returns. Default stays ATOMIC (one committed step per gesture):
    schedule-forcing drivers deliberately fire between unsettled states.
    Opt in via the constructor (`Application(auto_settle=True)`) or, for a
    generated Application whose __init__ only takes `seed`, via a class
    attribute on the host subclass (`class MyHost(App): auto_settle = True`)
    or plain instance assignment -- no emit change needed."""

    def __init__(self, seed: int = 0, auto_settle: bool | None = None) -> None:
        if type(self).p is not Application.p:
            raise TypeError(
                f"{type(self).__name__} defines an attribute 'p', which would "
                "silently shadow the open-port namespace (backlog 249 Part B); "
                "rename the conflicting member"
            )
        self._seed = seed
        self._rng = random.Random(seed)
        if auto_settle is not None:
            self.auto_settle = auto_settle
        self._components: dict[str, Component] = {}
        self._channels: list[Channel] = []
        self._buffers: list[Buffer] = []
        self._synth_actions: list[SynthAction] = []
        self._invariants: list[tuple[str, Callable[[Application], bool]]] = []
        self._last_action: str | None = None
        self._port_namespace: PortNamespace | None = None
        self._port_ns_sig: tuple | None = None
        # The host event queue (backlog 252): every emission on an open
        # out-port appends a HostEvent in commit order. Host observation
        # history, not model state -- snapshot()/restore() never touch it.
        # Subsumes 249 C's per-port drain buffers (drain() reads it) and
        # feeds the trigger handlers (`_out_subscribers` + the generated
        # `on_<port>_<Tag>` named form) at settle.
        self._event_queue: list[HostEvent] = []
        self._event_seq = 0
        self._out_subscribers: dict[tuple[str, str], list] = {}
        # Cascade tracking (the causal wake). `_cascade_background` is the
        # set of action names already enabled when the current cascade's
        # first gesture arrived -- background, not consequences; None means
        # no cascade is open. `_cascade_visited` is the settle-scoped
        # visited-state set behind the oscillation stop.
        self._cascade_background: set[str] | None = None
        self._cascade_visited: set | None = None
        self._settling = False
        self._commit_depth = 0
        self.last_settle: SettleVerdict | None = None

    # ----- Registration / binding ---------------------------------

    def _register(self, alias: str, component: Component) -> None:
        """Register a component under `alias`, handling composites.

        Two-pass contract:

          Pass 1: walk the composite tree and register every
                  component in `_components` before any internal
                  bindings replay. That way fan-in / fan-out /
                  channel / observe calls in the bindings can
                  reference any sub-component by reference without
                  worrying about registration order.

          Pass 2: walk the same tree and replay each composite's
                  `_internal_bindings` via the standard _channel /
                  _observe / _fanout / _fanin API. Bindings from
                  deeper composites replay first (post-order) so
                  any reference to an already-registered leaf
                  resolves cleanly.

        Compound alias collisions (e.g. `O.Mid.Leaf` and `O_Mid.Leaf`
        both normalizing to `O_Mid_Leaf` for hook dispatch) raise
        immediately.
        """
        self._register_tree(alias, component)
        self._replay_bindings_tree(component)

    def _unregister(self, alias: str) -> None:
        """Remove `alias` and its composite sub-tree from the registry,
        clearing each component's back-reference. Inverse of
        `_register`. Backlog 217 C2: lets a test or notebook free an
        alias and re-register without rebuilding the Application.

        Note this drops the alias from `_components`; it does not unwind
        channels/buffers/invariants that referenced it. For a full
        teardown use `reset()`.
        """
        component = self._components.get(alias)
        if component is None:
            raise KeyError(f"alias {alias!r} is not registered")
        for sub_suffix, sub_comp in (
            getattr(component, "_sub_components", {}) or {}
        ).items():
            self._unregister(f"{alias}.{sub_suffix}")
        del self._components[alias]
        component._app = None
        component._alias = ""

    def reset(self) -> None:
        """Clear all registration state, restoring the Application to its
        freshly-constructed shape (seed/RNG preserved). Backlog 217 C2:
        the register cycle is otherwise unrecoverable -- a duplicate
        alias raises with no way back, so test reuse and notebook
        re-eval would need a brand-new Application. `reset()` makes the
        same instance reusable."""
        for component in self._components.values():
            component._app = None
            component._alias = ""
        self._components = {}
        self._channels = []
        self._buffers = []
        self._synth_actions = []
        self._invariants = []
        self._last_action = None
        self._event_queue = []
        self._event_seq = 0
        self._out_subscribers = {}
        self._cascade_background = None
        self._cascade_visited = None
        self.last_settle = None

    def _register_tree(self, alias: str, component: Component) -> None:
        """Pass 1: register `component` and recurse into its
        `_sub_components` with compound aliases. Does not apply
        internal bindings."""
        if alias in self._components:
            raise ValueError(
                f"alias {alias!r} already registered"
            )
        # Compound-alias hook-name collision check. "O.Mid.Leaf" and
        # "O_Mid.Leaf" both normalize to "O_Mid_Leaf"; that would
        # collide in hook dispatch. Very unlikely with typical
        # CamelCase aliases, but cheap to catch.
        normalized = alias.replace(".", "_")
        for existing in self._components:
            if existing.replace(".", "_") == normalized and existing != alias:
                raise ValueError(
                    f"alias {alias!r} normalizes to hook-name "
                    f"{normalized!r} which collides with already-"
                    f"registered alias {existing!r}"
                )
        self._components[alias] = component
        component._app = self
        component._alias = alias
        for sub_suffix, sub_comp in (
            getattr(component, "_sub_components", {}) or {}
        ).items():
            self._register_tree(f"{alias}.{sub_suffix}", sub_comp)

    def _replay_bindings_tree(self, component: Component) -> None:
        """Pass 2: depth-first replay of `_internal_bindings`.
        Deepest composites bind first; outer composites bind after
        their sub-trees are wired up."""
        for sub_comp in (
            getattr(component, "_sub_components", {}) or {}
        ).values():
            self._replay_bindings_tree(sub_comp)
        for binding in getattr(component, "_internal_bindings", ()) or ():
            kind = binding["kind"]
            if kind == "channel":
                self._channel(
                    binding["a_comp"], binding["a_port"],
                    binding["b_comp"], binding["b_port"],
                )
            elif kind == "observe":
                self._observe(
                    binding["observer"], binding["port"],
                    binding["target"],
                )
            elif kind == "fan":
                # Direction-agnostic fan. Runtime dispatches to
                # _fanout vs _fanin via class-attr port directions.
                self._fan(binding["endpoints"], owner=component)
            elif kind == "invariant":
                # Composite MODULE invariants register with the
                # outer Application the same way Application-level
                # invariants do. Predicate takes `app` and reads
                # state through app._components lookups.
                self._register_invariant(
                    binding["id"], binding["predicate"],
                )
            elif kind == "buffer":
                # Register a Buffer (already constructed in the
                # composite's __init__ under self._buffers_by_name)
                # with the Application for snapshot/restore. Runs
                # before buffer_enqueue / buffer_dequeue bindings so
                # those can reference it by name.
                buf = component._buffers_by_name[binding["name"]]
                self._buffers.append(buf)
            elif kind == "internal_channel":
                # Allocate a Channel for a composite-internal port with
                # no external binder: one endpoint of a buffered path,
                # OR a composite-owned dead-letter (a dangling channel,
                # backlog 236). `sub` follows the buffer-endpoint
                # convention -- "" means the composite's OWN port (a
                # folded inline actor), otherwise a sub-component --
                # resolved the same way `_resolve_endpoint` does so an
                # inline-actor port (sub="") lands on the composite
                # itself, not a missing `_sub_components[""]`.
                port = binding["port"]
                sub = _resolve_endpoint(component, (binding["sub"], port))
                ch = Channel(
                    name=f"{sub._alias}.{port}"
                )
                sub._port_channels[port] = ch
                self._channels.append(ch)
            elif kind == "buffer_enqueue":
                # Synthesize an enqueue action from (source port) to
                # (buffer). `source` is (alias, port); alias="" means
                # the composite itself, otherwise a sub-component of
                # this composite. Resolved to Component references
                # now; the SynthAction looks up the Channel lazily.
                source_comp = _resolve_endpoint(
                    component, binding["source"],
                )
                buf = component._buffers_by_name[binding["buffer"]]
                synth = BufferEnqueue(
                    binding["action"],
                    source_comp, binding["source"][1],
                    buf,
                )
                synth.component = component
                self._synth_actions.append(synth)
            elif kind == "buffer_dequeue":
                # Synthesize a dequeue action from (buffer) to
                # (target port). Same endpoint convention as
                # buffer_enqueue.
                buf = component._buffers_by_name[binding["buffer"]]
                target_comp = _resolve_endpoint(
                    component, binding["target"],
                )
                synth = BufferDequeue(
                    binding["action"],
                    buf,
                    target_comp, binding["target"][1],
                )
                synth.component = component
                self._synth_actions.append(synth)
            elif kind == "fanout":
                # Atomic N-way broadcast inside a composite MODULE
                # (backlog 103). Uses the same API as application-
                # level fan-out; the composite's sub-components are
                # already registered, so _fanout can install the
                # source + per-receiver channels.
                self._fanout(
                    binding["sender"], binding["sender_port"],
                    binding["receivers"],
                    owner=component,
                )
            elif kind == "fanin":
                # Atomic per-sender merge inside a composite MODULE
                # (backlog 103).
                self._fanin(
                    binding["senders"],
                    binding["receiver"], binding["receiver_port"],
                    owner=component,
                )
            elif kind == "fanout_buffer_append":
                # Buffered N-way broadcast (backlog 074.1 + 103).
                # Source is a port on the publisher; buffers are the
                # per-subscriber Seq buffers listed by name. Each
                # buffer was already registered by a preceding
                # `buffer` binding.
                source_comp = _resolve_endpoint(
                    component, binding["source"],
                )
                buffers = [
                    component._buffers_by_name[n]
                    for n in binding["buffers"]
                ]
                synth = FanOutBufferAppend(
                    binding["action"],
                    source_comp, binding["source"][1],
                    buffers,
                )
                synth.component = component
                self._synth_actions.append(synth)
            else:
                raise ValueError(
                    f"_internal_bindings: unknown kind {kind!r}"
                )

    def _stub_channel(
        self,
        comp: Component,
        port: str,
        state_constants: str = "shared",
    ) -> Channel:
        """Register a single-endpoint dead-letter channel for an
        unbound out-port (backlog 182). Source transitions whose `sends`
        target this port find the channel clear (state == 'NotSent')
        and fire normally; the send writes 'InFlight' but no peer
        receives. Mirrors what TLC does via its synthetic project-level
        variable for unbound out-ports.

        Sender-only registration -- no _in_ports/_out_ports
        direction resolution and no peer side.

        `state_constants` records which TLA state-constant family the
        compiler chose for this channel (backlog 201.3). It is stored
        on the Channel and read by `tools/runtime_state.to_tla_view`.

        Backlog 225: anchor the dead-letter at the LEAF, mirroring the
        bound path (`_channel` resolves forwards before registering). For
        a composite that re-exports an inner out-port and leaves it unbound
        at the top, the Channel must live on the leaf's `_port_channels`
        -- that is where the inner send writes and where the projection /
        inherited invariant read it. Registering on the composite instead
        left the leaf send permanently disabled (no `_port_channels` entry
        -> `_channel_is_clear` False) and shadowed the live channel with a
        dead stub. No-op for a simple actor unbound port (no `_forwards`
        entry -> `_resolve_forward` returns the input unchanged)."""
        leaf_comp, leaf_port = self._resolve_forward(comp, port)
        name = f"{comp._alias}.{port}->(unbound)"
        ch = Channel(name=name, state_constants=state_constants)
        leaf_comp._port_channels[leaf_port] = ch
        self._channels.append(ch)
        return ch

    def _channel(
        self,
        comp_a: Component,
        port_a: str,
        comp_b: Component,
        port_b: str,
    ) -> Channel:
        """Point-to-point binding. Accepts the two endpoints in
        either order; resolves sender/receiver from each component's
        declared `_in_ports` / `_out_ports`.

        If either side is a composite port (present in that
        component's `_forwards` table), the binding is redirected
        to the forwarded (sub_component, sub_port) before resolving
        direction. Forwards chain recursively -- a two-layer
        composite's EXT_IN -> mid.MID_IN -> leaf.IN all lands on
        leaf.IN.
        """
        comp_a, port_a = self._resolve_forward(comp_a, port_a)
        comp_b, port_b = self._resolve_forward(comp_b, port_b)
        sender, out_port, receiver, in_port = self._resolve_direction(
            comp_a, port_a, comp_b, port_b,
        )
        name = (
            f"{sender._alias}.{out_port}->"
            f"{receiver._alias}.{in_port}"
        )
        ch = Channel(name=name)
        sender._port_channels[out_port] = ch
        receiver._port_channels[in_port] = ch
        self._channels.append(ch)
        return ch

    @staticmethod
    def _forwards_entry(entry) -> tuple[tuple["Component", str], ...]:
        """Normalize a `_forwards[port]` value to a tuple of
        `(Component, str)` endpoints.

        Accepts two shapes:
          - Bare `(comp, port_str)` pair (legacy, used by the
            hand-written composites in design/python-target/ and
            by pre-115 generated code).
          - Tuple of such pairs, emitted by the codegen post-115.

        Detects the bare-pair shape by checking the second element
        is a string; otherwise iterates.
        """
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[1], str)
        ):
            return (entry,)
        return tuple(entry)

    @staticmethod
    def _resolve_forward_all(
        comp: "Component", port: str,
    ) -> tuple[tuple["Component", str], ...]:
        """Walk a composite's `_forwards` table to the set of leaf
        `(Component, port)` endpoints. One external port may forward
        to multiple sub-components that share a source (typical for
        observe ports, backlog 115). Visited-set check catches
        malformed cycles. BFS over the forwards tree; returns at
        least one endpoint.
        """
        results: list[tuple["Component", str]] = []
        visited: set[tuple[int, str]] = set()
        queue: list[tuple["Component", str]] = [(comp, port)]
        while queue:
            c, p = queue.pop(0)
            forwards = getattr(c, "_forwards", None)
            if not forwards or p not in forwards:
                results.append((c, p))
                continue
            key = (id(c), p)
            if key in visited:
                raise ValueError(
                    f"_forwards cycle detected at "
                    f"{getattr(c, '_alias', '?')}.{p}"
                )
            visited.add(key)
            for endpoint in Application._forwards_entry(forwards[p]):
                queue.append(endpoint)
        return tuple(results)

    @staticmethod
    def _resolve_forward(
        comp: "Component", port: str,
    ) -> tuple["Component", str]:
        """Single-endpoint forward resolution. Wraps
        `_resolve_forward_all` and asserts exactly one leaf target.

        Channel / fan binding call sites expect a single-target
        forward -- an external in/out port forwarding to two
        sub-components is a fan, which is bound separately. This
        helper keeps those call sites simple and surfaces the
        forbidden-shape case as an explicit error.
        """
        endpoints = Application._resolve_forward_all(comp, port)
        if len(endpoints) != 1:
            raise ValueError(
                f"_forwards[{getattr(comp, '_alias', '?')}.{port}] "
                f"resolves to {len(endpoints)} endpoints; "
                f"channel / fan call sites require exactly one. "
                f"Only observe-port forwards allow N>1."
            )
        return endpoints[0]

    def _channel_phase(self, comp: "Component", port: str) -> str:
        """The TLA-view phase of the channel bound to `comp.port`. Resolves any composite
        forward to the leaf the `Channel` is actually installed on, then encodes the
        lifecycle the way TLA spells it: `NotSent`/`Delivered` -> `Ch_NotSent`/`Ch_Delivered`,
        else the in-flight carrier (a typed tag, or `Ch_InFlight` for the untyped phase).

        Shared by the App `_tla_` channel-VIEW properties AND App-scope invariant
        channel-port reads, so forwarding resolution + phase encoding live in ONE place
        (here), never duplicated at emit time."""
        c, p = self._resolve_forward(comp, port)
        ch = c._port_channels[p]
        if ch.state == "NotSent":
            return "Ch_NotSent"
        if ch.state == "Delivered":
            return "Ch_Delivered"
        return ch.tag if ch.tag is not None else "Ch_InFlight"

    @staticmethod
    def _resolve_direction(
        comp_a: Component, port_a: str,
        comp_b: Component, port_b: str,
    ) -> tuple[Component, str, Component, str]:
        """Figure out which (comp, port) is the sender and which is
        the receiver. Uses the components' class-level port
        declarations. Raises if the two ports aren't one-in /
        one-out."""
        a_is_out = port_a in comp_a._out_ports
        a_is_in = port_a in comp_a._in_ports
        b_is_out = port_b in comp_b._out_ports
        b_is_in = port_b in comp_b._in_ports
        if a_is_out and b_is_in:
            return comp_a, port_a, comp_b, port_b
        if b_is_out and a_is_in:
            return comp_b, port_b, comp_a, port_a
        raise ValueError(
            f"_channel: cannot resolve direction for "
            f"{comp_a._alias}.{port_a} <-> {comp_b._alias}.{port_b} "
            f"(neither side is a clean out/in pair)"
        )

    def _observe(
        self,
        observer: Component,
        port_name: str,
        target: Component,
    ) -> None:
        """Bind an observe port to the component being observed.
        Guards on every observer read `target.state` directly.

        If `observer.port_name` is a forwarded composite port, the
        bind lands on every internal sub-component that shares the
        port (backlog 115). A composite whose CoinCup binds WATCH_GM
        to both inner coins resolves to two leaf endpoints; each
        coin gets its own `_observed['WATCH_GM']` entry pointing at
        the same target. Single-target observe forwards continue to
        work -- the tuple-of-endpoints shape collapses to length 1.
        The target side never forwards (we observe a concrete
        component's state, not a composite).
        """
        endpoints = self._resolve_forward_all(observer, port_name)
        for endpoint_comp, endpoint_port in endpoints:
            if endpoint_port not in endpoint_comp._observe_ports:
                raise ValueError(
                    f"{endpoint_comp._alias}.{endpoint_port} "
                    f"is not an observe port"
                )
            endpoint_comp._observed[endpoint_port] = target

    def _fanout(
        self,
        sender: Component,
        out_port: str,
        receivers: list[tuple[Component, str]],
        owner: "Component | None" = None,
    ) -> None:
        """Broadcast: one sender, N receivers. Sender has one source
        channel; each receiver has its own. A synthesized `FanOut`
        action copies source -> all targets atomically when enabled.

        Endpoints may be composite ports; `_resolve_forward` chains
        through to the real leaf sub-component + port before the
        source/target channels are created.

        `owner` tags the synthesized action with its owning composite
        so a component-level `on_<action>` hook fires on the composite
        subclass in `_commit_synth`. APPLICATION-mode callers leave
        `owner=None` (the fan belongs to the Application itself).
        """
        sender, out_port = self._resolve_forward(sender, out_port)
        self._require_out_port(sender, out_port)
        source = Channel(name=f"{sender._alias}.{out_port}->(fanout)")
        sender._port_channels[out_port] = source
        self._channels.append(source)

        targets: list[Channel] = []
        for receiver, in_port in receivers:
            receiver, in_port = self._resolve_forward(receiver, in_port)
            self._require_in_port(receiver, in_port)
            ch_name = (
                f"{sender._alias}.{out_port}->"
                f"{receiver._alias}.{in_port}"
            )
            ch = Channel(name=ch_name)
            receiver._port_channels[in_port] = ch
            self._channels.append(ch)
            targets.append(ch)

        action_name = f"{sender._alias}.{out_port}_fanout"
        fa = FanOut(action_name, source, targets)
        if owner is not None:
            fa.component = owner
        self._synth_actions.append(fa)

    def _fan(
        self,
        endpoints: list[tuple[Component, str]],
        owner: "Component | None" = None,
    ) -> None:
        """Direction-agnostic fan binding. Given N endpoints, look up
        each port's direction on its component's class attrs, group
        by direction, and dispatch to `_fanout` or `_fanin`:

          1 out + N>=2 ins  -> fan-out (out is the publisher)
          1 in  + N>=2 outs -> fan-in  (in is the receiver)

        Composites don't know at codegen time which side is the 1 vs
        the N (the IR has a shared channel variable and 3+ endpoint
        with_mappings, but no direction metadata). The runtime
        resolves it at bind time via class attrs.

        `owner` is passed through to `_fanout` / `_fanin` so the
        synthesized fan action gets tagged with its owning composite
        for component-level hook dispatch.
        """
        ins: list[tuple[Component, str]] = []
        outs: list[tuple[Component, str]] = []
        for comp, port in endpoints:
            comp_, port_ = self._resolve_forward(comp, port)
            if port_ in comp_._in_ports:
                ins.append((comp_, port_))
            elif port_ in comp_._out_ports:
                outs.append((comp_, port_))
            else:
                raise ValueError(
                    f"_fan: {comp_._alias}.{port_} is neither in nor out"
                )
        if len(outs) == 1 and len(ins) >= 2:
            sender, out_port = outs[0]
            self._fanout(sender, out_port, ins, owner=owner)
        elif len(ins) == 1 and len(outs) >= 2:
            receiver, in_port = ins[0]
            self._fanin(outs, receiver, in_port, owner=owner)
        else:
            raise ValueError(
                f"_fan: can't dispatch {len(outs)} senders + "
                f"{len(ins)} receivers"
            )

    def _fanin(
        self,
        senders: list[tuple[Component, str]],
        receiver: Component,
        in_port: str,
        owner: "Component | None" = None,
    ) -> None:
        """Merge: N senders, one receiver. Receiver has one shared
        inbound channel; each sender has its own source. A
        `FanInMerge` action per sender picks that source into the
        shared inbound.

        Endpoints may be composite ports; `_resolve_forward` chains
        through to the real leaf sub-component + port.

        `owner` tags each synthesized merge action with its owning
        composite so a component-level `on_<action>` hook fires on
        the composite subclass. APPLICATION-mode callers leave
        `owner=None` (the merges belong to the Application itself).
        """
        receiver, in_port = self._resolve_forward(receiver, in_port)
        self._require_in_port(receiver, in_port)
        dest = Channel(name=f"(fanin)->{receiver._alias}.{in_port}")
        receiver._port_channels[in_port] = dest
        self._channels.append(dest)

        for sender, out_port in senders:
            sender, out_port = self._resolve_forward(sender, out_port)
            self._require_out_port(sender, out_port)
            src_name = (
                f"{sender._alias}.{out_port}->"
                f"{receiver._alias}.{in_port}"
            )
            source = Channel(name=src_name)
            sender._port_channels[out_port] = source
            self._channels.append(source)
            merge_name = f"{sender._alias}.{out_port}_fanin_merge"
            fim = FanInMerge(merge_name, source, dest)
            if owner is not None:
                fim.component = owner
            self._synth_actions.append(fim)

    def _fanout_buffered(
        self,
        sender: Component,
        out_port: str,
        receivers: list[tuple[Component, str]],
        depth: int,
        append_action: str,
        drain_actions: list[str],
        buffer_names: list[str],
        owner: "Component | None" = None,
    ) -> None:
        """Buffered N-way broadcast (backlog 074.1 + 077). Allocates
        the publisher's distribution channel, one per-receiver
        target channel, and one Seq buffer per receiver. Synthesizes
        one `FanOutBufferAppend` (atomic N-way enqueue: source ->
        every buffer) plus one `BufferDequeue` per receiver (pops
        its buffer head into the receiver's input channel).

        Same shape supports single-link bufferDepth (`receivers`
        length 1): the single buffer's enqueue drains into the
        receiver's port.

        `owner` tags each synthesized action with its optional
        owning composite so component-level hooks fire there too.
        APPLICATION-mode callers leave `owner=None` -- the synth
        actions belong to the Application itself.
        """
        assert len(receivers) == len(drain_actions) == len(buffer_names), (
            "receivers / drain_actions / buffer_names must align"
        )
        sender, out_port = self._resolve_forward(sender, out_port)
        self._require_out_port(sender, out_port)
        source = Channel(name=f"{sender._alias}.{out_port}")
        sender._port_channels[out_port] = source
        self._channels.append(source)

        buffers: list[Buffer] = []
        resolved_receivers: list[tuple[Component, str]] = []
        for (recv, in_port), buf_name in zip(receivers, buffer_names):
            recv, in_port = self._resolve_forward(recv, in_port)
            self._require_in_port(recv, in_port)
            ch = Channel(
                name=f"{sender._alias}.{out_port}->{recv._alias}.{in_port}"
            )
            recv._port_channels[in_port] = ch
            self._channels.append(ch)
            buf = Buffer(name=buf_name, depth=depth)
            self._buffers.append(buf)
            buffers.append(buf)
            resolved_receivers.append((recv, in_port))

        append = FanOutBufferAppend(
            append_action, sender, out_port, buffers,
        )
        if owner is not None:
            append.component = owner
        self._synth_actions.append(append)

        for (recv, in_port), buf, drain_name in zip(
            resolved_receivers, buffers, drain_actions,
        ):
            drain = BufferDequeue(drain_name, buf, recv, in_port)
            if owner is not None:
                drain.component = owner
            self._synth_actions.append(drain)

    def _register_invariant(
        self,
        inv_id: str,
        predicate_fn: Callable[[Application], bool],
    ) -> None:
        self._invariants.append((inv_id, predicate_fn))

    @staticmethod
    def _require_out_port(comp: Component, port: str) -> None:
        if port not in comp._out_ports:
            raise ValueError(f"{comp._alias}.{port} is not an out-port")

    @staticmethod
    def _require_in_port(comp: Component, port: str) -> None:
        if port not in comp._in_ports:
            raise ValueError(f"{comp._alias}.{port} is not an in-port")

    # ----- Host port surface (backlog 249 Part B) ------------------

    @property
    def p(self) -> PortNamespace:
        """The open-port namespace: one callable `OpenInPort` per open
        input port. Rebuilt when the component registry or channel set
        changes (a driver popping a sibling opens its peer's port), so
        the view always reflects the live boundary."""
        sig = (tuple(self._components), len(self._channels))
        if self._port_namespace is None or self._port_ns_sig != sig:
            self._port_namespace = self._build_port_namespace()
            self._port_ns_sig = sig
        return self._port_namespace

    def _build_port_namespace(self) -> PortNamespace:
        # Walk in registration order (outermost composite before its
        # subs), resolving each port forward to its leaf endpoint and
        # claiming each open leaf endpoint ONCE -- so a composite
        # interface port and the leaf port it forwards to are one entry,
        # named for the outermost (the boundary the binding would target).
        # In-ports become callable OpenInPorts; out-ports observable
        # OpenOutPorts (backlog 249 C). One shared name pool.
        entries: list[tuple[str, str, Component, str, bool]] = []
        claimed: set[tuple[int, str]] = set()
        for alias, comp in self._components.items():
            for port, is_in in ([(p, True) for p in comp._in_ports]
                                + [(p, False) for p in comp._out_ports]):
                leaf_comp, leaf_port = self._resolve_forward(comp, port)
                if leaf_comp._port_channels.get(leaf_port) is not None:
                    continue  # bound -- faces a sibling, not the host
                key = (id(leaf_comp), leaf_port)
                if key in claimed:
                    continue
                claimed.add(key)
                entries.append((alias, port, leaf_comp, leaf_port, is_in))
        # Flat name when globally unique; alias-qualified when two
        # instances expose the same port name; loud failure if names
        # still collide. Deterministic from (alias, port) -- IR names.
        flat_counts: dict[str, int] = {}
        for _, port, _, _, _ in entries:
            n = _surface_name(port)
            flat_counts[n] = flat_counts.get(n, 0) + 1
        ports: dict[str, Any] = {}
        for alias, port, leaf_comp, leaf_port, is_in in entries:
            n = _surface_name(port)
            if flat_counts[n] > 1:
                n = _surface_name(alias, port)
            if n in ports:
                raise ValueError(
                    f"open-port surface name collision: {n!r} (from "
                    f"{alias}.{port}) -- already claimed by {ports[n]!r}"
                )
            cls = OpenInPort if is_in else OpenOutPort
            ports[n] = cls(self, leaf_comp, leaf_port, f"{alias}.{port}")
        return PortNamespace(ports)

    # ----- Enabled-set enumeration --------------------------------

    def _enabled(self) -> list[tuple[str, Component | None, Transition | SynthAction]]:
        """Every action ready to fire right now. Component transitions
        and synthesized actions walked uniformly. The scheduler picks
        from this list; fire() validates against it."""
        out: list[tuple[str, Component | None, Transition | SynthAction]] = []
        for alias, comp in self._components.items():
            for t in comp._transitions:
                if getattr(comp, t.state_field) != t.from_state:
                    continue
                if not self._transition_enabled(comp, t):
                    continue
                out.append((f"{alias}.{t.name}", comp, t))
        for sa in self._synth_actions:
            if sa.is_enabled():
                out.append((sa.name, None, sa))
        return out

    def _transition_enabled(self, comp: Component, t: Transition) -> bool:
        """Guard + per-port channel preconditions. Returns True iff
        firing this transition right now would succeed.

        Backlog 201.2: guard predicates are pure functions over actor
        state and must evaluate cleanly to a bool. Any exception is
        a bug; let it propagate. The prior `_safe_guard` helper
        swallowed all exceptions as "not enabled", which silently
        disabled transitions whose guards threw -- backlog 201.2's
        recv-kind alias bug spent months as `tlc-py > 0` cross-target
        gap behind that swallow. Distinguishing "exception means
        False" from "exception means error" is exactly the
        defensive-fallback pattern 198 retired."""
        if t.guard_fn is not None and not bool(t.guard_fn(comp)):
            return False
        if (
            t.recv_port is not None
            and not self._is_open_input(comp, t.recv_port)
            and not self._channel_has_inflight(comp, t.recv_port, t.recv_tag)
        ):
            return False
        for s in t.sends:
            if self._is_open_output(comp, s.port):
                continue  # most-general receiver: output never blocks (249 B2)
            if not self._channel_is_clear(comp, s.port):
                return False
        return True

    @staticmethod
    def _channel_has_inflight(comp: Component, port: str, tag: str | None) -> bool:
        """Receive guard: channel must be InFlight, and either the
        transition requires a specific tag (must match exactly) or
        it's a wildcard receive (recv_tag=None, matches any tag).

        Backlog 088: untagged receive on a typed channel is a
        wildcard -- TLA+'s `channel = Ch_InFlight` guard is
        satisfied regardless of the tag value stored in the
        channel. Without wildcard semantics, an untagged module
        composed into a typed parent deadlocks in Python where
        TLC would progress."""
        ch = comp._port_channels.get(port)
        if ch is None:
            return False
        if ch.state != "InFlight":
            return False
        if tag is None:
            return True
        return ch.tag == tag

    @staticmethod
    def _channel_is_clear(comp: Component, port: str) -> bool:
        ch = comp._port_channels.get(port)
        if ch is None:
            return False
        return ch.state in ("NotSent", "Delivered")

    @staticmethod
    def _is_open_output(comp: Component, port: str) -> bool:
        """An OPEN output port: declared in `_out_ports`, no channel after
        `_bind` (backlog 249 B2: an `@host`-declared out-port gets no stub --
        the channel pass minted no node). The environment is the most-general
        receiver: the send fires freely and never wedges, mirroring the TLA
        `<Port>Open` collapse. Distinct from the 182 dead-letter stub (an
        UNMARKED dangle), which keeps a Channel and its lifecycle."""
        return port in comp._out_ports and comp._port_channels.get(port) is None

    @staticmethod
    def _is_open_input(comp: Component, port: str) -> bool:
        """An OPEN input port: declared in `_in_ports` but left unbound (no channel
        after `_bind`). Backlog 249. E009 (`analyze_pass`) forbids an unbound in-port
        on a *declared* component, so this only ever arises by design -- a standalone
        leaf / composite-interface port that is the system's environment boundary. The
        environment (a most-general sender, or, in an app, the host) may deliver any
        accepted message whenever it likes, so a receive on such a port is enabled on
        its guard alone. This mirrors the TLA emitter, which drops the channel for an
        unwired port and emits the receive as a free, guard-only local transition --
        restoring cross-target reachability parity (and powering host-driven `port(tag)`)."""
        return port in comp._in_ports and comp._port_channels.get(port) is None

    # ----- Fire (explicit) / step (scheduler) ---------------------

    def fire(self, qualified_name: str) -> TransitionEvent:
        """Force a named action. Raises DisabledActionError if not
        enabled. Same commit path as `step()`. A host-initiated fire is a
        GESTURE (backlog 252): it opens a cascade if none is pending, and
        in Node mode (`auto_settle=True`) settle+dispatch fold in before
        it returns."""
        self._begin_cascade()
        sa = self._find_synth(qualified_name)
        if sa is not None:
            if not sa.is_enabled():
                raise DisabledActionError(
                    qualified_name, "-", "synth guard not satisfied"
                )
            event = self._commit_synth(sa)
            self._maybe_auto_settle()
            return event
        alias, comp, t = self._resolve_component_action(qualified_name)
        if getattr(comp, t.state_field) != t.from_state:
            raise DisabledActionError(
                qualified_name, getattr(comp, t.state_field),
                f"requires from_state={t.from_state!r}",
            )
        if not self._transition_enabled(comp, t):
            raise DisabledActionError(
                qualified_name, getattr(comp, t.state_field),
                "guard or channel precondition not satisfied",
            )
        event = self._commit(alias, comp, t)
        self._maybe_auto_settle()
        return event

    def step(self) -> TransitionEvent | None:
        """Scheduler pick. Returns the event that fired, or None if
        nothing was enabled."""
        enabled = self._enabled()
        if not enabled:
            self._call_hook("on_step", None)
            return None
        _, comp, thing = self._rng.choice(enabled)
        if isinstance(thing, SynthAction):
            event = self._commit_synth(thing)
        else:
            event = self._commit(comp._alias, comp, thing)
        self._call_hook("on_step", event)
        return event

    def _find_synth(self, name: str) -> SynthAction | None:
        for sa in self._synth_actions:
            if sa.name == name:
                return sa
        return None

    def _resolve_component_action(
        self, qualified_name: str,
    ) -> tuple[str, Component, Transition]:
        # Composite components register with compound aliases like
        # "O.Mid.Leaf". We rpartition on the LAST dot so the alias
        # stays intact and the trailing bit is the transition name.
        # Transition names never contain dots (convention).
        alias, _, action = qualified_name.rpartition(".")
        comp = self._components.get(alias)
        if comp is None:
            raise DisabledActionError(
                qualified_name, "?",
                f"no component registered with alias {alias!r}",
            )
        for t in comp._transitions:
            if t.name == action:
                return alias, comp, t
        raise DisabledActionError(
            qualified_name, comp.state,
            f"component has no transition named {action!r}",
        )

    # ----- The host reactive loop (backlog 252) --------------------
    #
    # A host gesture commits; the runtime SETTLES the world (pump: the
    # causal wake), during which every component->host trigger firing
    # lands on the event queue; at settle the queue DRAINS through the
    # host's trigger handlers against the settled state; handler
    # reactions are new gestures whose consequences are next-round
    # tasks; repeat until both the world and the queue are quiet.
    # Soundness untouched: deferral changes only WHEN host code runs --
    # every fire stays its own atomic committed verified step.

    def _begin_cascade(self) -> None:
        """Open a cascade at the first gesture since the last settled
        moment: capture the background (every action enabled BEFORE the
        gesture -- already-enabled moves are background, not consequences)
        and seed the visited-state set with the pre-gesture world."""
        if self._cascade_background is None:
            self._cascade_background = {
                name for name, _, _ in self._enabled()
            }
            self._cascade_visited = {self._fingerprint()}

    def _end_cascade(self) -> None:
        self._cascade_background = None
        self._cascade_visited = None

    def _fingerprint(self) -> tuple:
        """Hashable projection of spec-level state, for the visited-state
        oscillation stop and the (state, queue) livelock check."""
        return tuple(sorted(
            self.state_snapshot().items(), key=lambda kv: kv[0],
        ))

    def _wake(self) -> list[tuple[str, Component | None, Transition | SynthAction]]:
        """The causal wake: enabled actions that BECAME enabled during
        this cascade. Excludes the background (enabled before the
        gesture), the perimeter (receives on open ports -- those are the
        host's/environment's moves to make, the blind-step() defect), and
        pure stutter self-loops (from==to, no recv/sends/assigns/effect:
        TLC deadlock furniture that commits nothing)."""
        background = self._cascade_background or set()
        out = []
        for name, comp, thing in self._enabled():
            if name in background:
                continue
            if isinstance(thing, Transition):
                if (thing.recv_port is not None
                        and self._is_open_input(comp, thing.recv_port)):
                    continue
                if (thing.from_state == thing.to_state
                        and thing.recv_port is None
                        and not thing.sends and not thing.assigns
                        and thing.effect_fn is None):
                    continue
            out.append((name, comp, thing))
        return out

    def pump(self) -> PumpReport:
        """The world's turn: fire the causal wake to exhaustion, one
        committed step at a time (first enabled wake action each
        iteration -- registration order, deterministic). Terminates: the
        wake shrinks toward the background, and the visited-state stop
        catches cycles (finite state space => guaranteed). Ends the
        cascade; the next gesture starts fresh."""
        self._begin_cascade()
        report = self._pump_world()
        self._end_cascade()
        return report

    def _pump_world(self) -> PumpReport:
        """Settle-internal pump: same loop as `pump()` but keeps the
        cascade open (settle's rounds share one background + one
        visited-state set, per 252 decision 1)."""
        events: list[TransitionEvent] = []
        oscillation: str | None = None
        visited = self._cascade_visited
        visited.add(self._fingerprint())
        while True:
            wake = self._wake()
            if not wake:
                break
            name, comp, thing = wake[0]
            if isinstance(thing, SynthAction):
                events.append(self._commit_synth(thing))
            else:
                events.append(self._commit(comp._alias, comp, thing))
            fp = self._fingerprint()
            if fp in visited:
                oscillation = (
                    f"world state revisited after {name} -- "
                    f"the world oscillates here"
                )
                break
            visited.add(fp)
        return PumpReport(events=tuple(events), oscillation=oscillation)

    def settle(self) -> SettleVerdict:
        """Settle the world, then notify (Node-style): pump the causal
        wake quiet, drain the event queue through the host's trigger
        handlers, repeat -- handler gestures commit atomically but their
        consequences are NEXT-ROUND tasks (bounded stack, deterministic
        drain order). Returns the verdict (also passed to the `on_settle`
        hook and kept on `self.last_settle`).

        Called from inside a running settle (a handler) or mid-commit (a
        hook), it is a no-op returning `deferred=True`: the outer loop
        owns the cascade."""
        if self._settling or self._commit_depth > 0:
            return SettleVerdict(
                steps=(), dispatched=(), rounds=0, oscillation=None,
                livelock=False, enabled=(), deferred=True,
            )
        self._settling = True
        try:
            self._begin_cascade()
            steps: list[TransitionEvent] = []
            dispatched: list[HostEvent] = []
            rounds = 0
            oscillation: str | None = None
            livelock = False
            seen_configs: set = set()
            while True:
                report = self._pump_world()
                steps.extend(report.events)
                oscillation = oscillation or report.oscillation
                batch = [e for e in self._event_queue if not e.dispatched]
                if not batch:
                    break
                config = (
                    self._fingerprint(),
                    tuple((e.key, e.tag) for e in batch),
                )
                if config in seen_configs:
                    livelock = True
                    break
                seen_configs.add(config)
                rounds += 1
                handler_names = {
                    obj._key: name for name, obj in self.p
                    if isinstance(obj, OpenOutPort)
                }
                for e in batch:
                    e.dispatched = True
                    dispatched.append(e)
                    self._dispatch_host_event(e, handler_names)
            self._end_cascade()
            verdict = SettleVerdict(
                steps=tuple(steps),
                dispatched=tuple(dispatched),
                rounds=rounds,
                oscillation=oscillation,
                livelock=livelock,
                enabled=tuple(sorted(n for n, _, _ in self._enabled())),
            )
            self.last_settle = verdict
            self._call_hook("on_settle", verdict)
            return verdict
        finally:
            self._settling = False

    def _dispatch_host_event(
        self, e: HostEvent, handler_names: dict[tuple[str, str], str],
    ) -> None:
        """Push one settled event through the host's trigger surface:
        the `on_drain` journaling hook, then anonymous subscribers
        (fn(tag)), then the generated-name handler -- `on_<port>_<Tag>`
        for a tagged event, `on_<port>` for an untagged one. The port
        segment is the SAME `_surface_name` projection (and collision
        rule) the `app.p` namespace uses; the tag is appended verbatim
        (tags are IR constants, already identifiers). Never a flat-name
        parse."""
        self._call_hook("on_drain", e)
        for fn in self._out_subscribers.get(e.key, ()):
            fn(e.tag)
        port_name = handler_names.get(e.key)
        if port_name is not None:
            handler = (f"on_{port_name}_{e.tag}" if e.tag is not None
                       else f"on_{port_name}")
            self._call_hook(handler)

    def _maybe_auto_settle(self) -> None:
        """Node mode's fold-in, fired after a gesture commits. Skipped
        inside a running settle (handler gestures cascade next round,
        never nest) and inside a commit (a hook-fired follow-on must not
        settle from the outer commit's hook phase, before its invariant
        re-check)."""
        if self.auto_settle and not self._settling and self._commit_depth == 0:
            self.settle()

    # ----- Atomic commit: apply -> hooks -> invariants ------------
    #
    # Per transitions.md: all mutations happen inside `_apply`
    # before any hook runs. Hooks see the post-state.

    def _commit(
        self, alias: str, comp: Component, t: Transition,
    ) -> TransitionEvent:
        # Backlog 252: open out-port emissions enqueue HostEvents inside
        # `_apply`; trigger handlers (subscribers + named on_<port>_<Tag>
        # methods) hear them at SETTLE, not here -- hooks watch steps,
        # trigger handlers hear settled events. The depth counter lets a
        # hook-fired follow-on fire() commit without tripping auto-settle
        # mid-hook-phase.
        self._commit_depth += 1
        try:
            event = self._apply(alias, comp, t)
            self._call_transition_hooks(alias, comp, event)
            self._evaluate_invariants()
        finally:
            self._commit_depth -= 1
        return event

    def _apply(
        self, alias: str, comp: Component, t: Transition,
    ) -> TransitionEvent:
        """Commit every mutation of this transition. Component state,
        channels, counter, effect_fn -- all before any hook sees a
        thing. Returns the event describing what just happened."""
        from_state = getattr(comp, t.state_field)
        tag: str | None = None
        # Simultaneous-assign semantics (backlog 246). A transition's
        # assigns mirror TLA+ primes: every RHS reads the *pre-state*
        # (unprimed values). Evaluate them all here -- before `_apply`
        # mutates state_field, channel state/tag, or `_msg_in_flight`
        # below -- so no RHS can observe a write made by this same
        # transition (another assign's result, the new control state, a
        # channel it just drove, or a captured carrier). This is what
        # makes the apply both pre-state-faithful and order-independent
        # (the tell of a correct simultaneous implementation); the writes
        # happen after, where the read order no longer matters.
        new_vals = [(var_name, value_fn(comp)) for var_name, value_fn in t.assigns]
        # MessageSet enforcement: if the component declares a
        # non-empty _message_set (mirroring TLA+'s tagConstants +
        # `\\in MessageSet` guards), any tagged send must use a
        # value from that set. Untagged (None) always OK -- matches
        # TLA+'s Ch_InFlight untagged carrier semantics.
        for s in t.sends:
            if s.tag is not None and comp._message_set \
                    and s.tag not in comp._message_set:
                raise MessageSetViolation(
                    alias, s.port, s.tag, comp._message_set,
                )
        setattr(comp, t.state_field, t.to_state)
        if t.recv_port is not None:
            ch = comp._port_channels.get(t.recv_port)
            if ch is None:
                # Open input port (backlog 249): no channel to advance -- the
                # environment/host supplied the message out of band. The transition's
                # own recv_tag IS the delivered tag (None for a wildcard receive).
                tag = t.recv_tag
            else:
                ch.state = "Delivered"
                tag = ch.tag
        sends_payload: list[tuple[str, str | None]] = []
        for s in t.sends:
            ch = comp._port_channels.get(s.port)
            resolved = s.tag
            sends_payload.append((s.port, resolved))
            if ch is None:
                # Open output port (backlog 249 B2/C): no channel to drive --
                # the environment (most-general receiver / the host) accepts
                # the message out of band. The emission is a component->host
                # trigger firing (backlog 252): enqueue it for the host's
                # drain()/trigger-handler surface (observation history, not
                # model state -- snapshot never touches it).
                self._event_queue.append(HostEvent(
                    seq=self._event_seq,
                    key=(comp._alias, s.port),
                    tag=resolved,
                ))
                self._event_seq += 1
                if tag is None:
                    tag = resolved
                continue
            ch.tag = resolved
            ch.state = "InFlight"
            # First outgoing slot wins for the event payload's `tag`
            # field when no recv contributed one. The "if tag is None"
            # check handles both ordering (recv before sends) and the
            # send-only case (no recv assigned tag).
            if tag is None:
                tag = ch.tag
        # Bound enforcement (backlog 257 Finding B): an integer assign outside
        # the variable's declared bound is the Python mirror of TLA+ rejecting
        # the state via TypeInvariant. Check all before writing any, so a
        # violation halts before leaving partial state behind.
        for var_name, value in new_vals:
            bnd = comp._typed_var_bounds.get(var_name)
            if bnd is not None:
                # A bound endpoint may be a str naming an option (e.g.
                # [0, "depth"]) -- resolve it per-instance, the same way
                # __init__ resolves option-name typed-var defaults.
                lo = comp._options[bnd[0]] if isinstance(bnd[0], str) else bnd[0]
                hi = comp._options[bnd[1]] if isinstance(bnd[1], str) else bnd[1]
                if not (lo <= value <= hi):
                    raise BoundViolation(alias, var_name, value, lo, hi)
        # Write the assign results computed against the pre-state above.
        # Targets are component data vars (independent of state_field and
        # channels), so the write order here is immaterial.
        for var_name, value in new_vals:
            setattr(comp, var_name, value)
        if t.effect_fn is not None:
            t.effect_fn(comp)
        qname = f"{alias}.{t.name}"
        self._last_action = qname
        return TransitionEvent(
            qualified_name=qname,
            action_name=t.name,
            component_alias=alias,
            from_state=from_state,
            to_state=t.to_state,
            tag=tag,
            sends=tuple(sends_payload),
        )

    def _commit_synth(self, sa: SynthAction) -> TransitionEvent:
        """Atomic commit for a synthesized action. Fires two
        app-level hooks: the per-action name (with dots normalized
        to underscores so compound-alias fan actions -- e.g.
        `c.P.OUT_fanout` -- have valid Python identifiers that
        `getattr` can resolve) and the catch-all `on_fire`. The
        event's `action_name` and `qualified_name` retain the raw
        `.`-separated form so trace logs read the original
        identifier.

        Synth actions belonging to a composite (buffer / fan
        bindings replayed from `_internal_bindings`) optionally
        carry a `component` attribute pointing at the owning
        composite; if present, a component-level `on_<action>`
        hook on the composite subclass also fires -- matches the
        real-transition hook dispatch.
        """
        self._commit_depth += 1
        try:
            sa.apply()
            self._last_action = sa.name
            event = TransitionEvent(
                qualified_name=sa.name,
                action_name=sa.name,
                component_alias="",
                from_state="",
                to_state="",
                tag=None,
            )
            hook_id = sa.name.replace(".", "_")
            self._call_hook(f"on_{hook_id}", event)
            self._call_hook("on_fire", event)
            owner = getattr(sa, "component", None)
            if owner is not None:
                self._call_hook_on(owner, f"on_{hook_id}", event)
            self._evaluate_invariants()
        finally:
            self._commit_depth -= 1
        return event

    # ----- Hook dispatch ------------------------------------------
    #
    # Dispatch by method name via getattr. A hook that isn't defined
    # is a no-op. The list of hooks fired per transition is explicit
    # in `_call_transition_hooks` so a reviewer sees it whole.

    def _call_transition_hooks(
        self, alias: str, comp: Component, event: TransitionEvent,
    ) -> None:
        """Fire every hook that applies to this transition, in a
        fixed order: exit, action, enter, catch-all. App-level hooks
        use alias-prefixed names; component-level hooks use bare
        names (no prefix). Users override whichever fits.

        Compound aliases (e.g. `O.Mid.Leaf` for components nested
        in composites) get their dots replaced with underscores in
        hook names so they're valid Python identifiers. User writes
        `def on_O_Mid_Leaf_Leaf_Handle(self, event)`.
        """
        alias_id = alias.replace(".", "_")
        app_hooks = [
            f"on_{alias_id}_exit_{event.from_state}",
            f"on_{alias_id}_{event.action_name}",
            f"on_{alias_id}_enter_{event.to_state}",
            "on_fire",
        ]
        comp_hooks = [
            f"on_exit_{event.from_state}",
            f"on_{event.action_name}",
            f"on_enter_{event.to_state}",
        ]
        for name in app_hooks:
            self._call_hook(name, event)
        for name in comp_hooks:
            self._call_hook_on(comp, name, event)

    def _call_hook(self, name: str, *args: Any) -> None:
        """Invoke `self.<name>(*args)` if defined; otherwise no-op."""
        method = getattr(self, name, None)
        if callable(method):
            method(*args)

    @staticmethod
    def _call_hook_on(target: Any, name: str, *args: Any) -> None:
        """Invoke `target.<name>(*args)` if defined; otherwise no-op."""
        method = getattr(target, name, None)
        if callable(method):
            method(*args)

    def on_invariant_violated(self, exc: InvariantViolation) -> None:
        """Default: raise. Subclasses override to log, alert, or
        continue (the explicit non-default behavior)."""
        raise exc

    # ----- Invariant evaluation -----------------------------------

    def _evaluate_invariants(self) -> None:
        """Per invariants.md: every registered predicate evaluates
        after every state change. Failures flow through
        `on_invariant_violated` -- default raises."""
        for inv_id, predicate in self._invariants:
            try:
                ok = predicate(self)
            except Exception as e:
                self.on_invariant_violated(InvariantViolation(
                    inv_id, f"evaluation raised {e!r}", self._last_action,
                ))
                continue
            if not ok:
                self.on_invariant_violated(InvariantViolation(
                    inv_id, self._state_summary(), self._last_action,
                ))

    def _state_summary(self) -> str:
        def _one(alias: str, comp: Component) -> str:
            # Backlog 242: a multi-inline composite has several FSMs; show each.
            extra = "".join(
                f", {alias}.{n}={getattr(comp, n)}" for n in comp._state_fields
            )
            return f"{alias}={comp.state}{extra}"
        return ", ".join(
            _one(alias, comp) for alias, comp in self._components.items()
        )

    # ----- Snapshot / restore / replay ----------------------------

    def state_snapshot(self) -> dict[str, Any]:
        """Flat dict projection of spec-level state. Keyed by
        `<Alias>.state`, `<Alias>.<typed_var>`, the channel's own
        `name` attribute, and a buffer's `name` attribute (for
        buffered ports). Used by `replay()` to compare against a
        recorded trace."""
        snap: dict[str, Any] = {}
        for alias, comp in self._components.items():
            snap[f"{alias}.state"] = comp.state
            for name in comp._typed_var_defaults:
                snap[f"{alias}.{name}"] = getattr(comp, name)
            # Backlog 242: a multi-inline composite's extra FSM states.
            for name in comp._state_fields:
                snap[f"{alias}.{name}"] = getattr(comp, name)
        for ch in self._channels:
            snap[ch.name] = (ch.state, ch.tag)
        for buf in self._buffers:
            snap[buf.name] = tuple(buf.items)
        return snap

    def replay(
        self,
        trace: list[tuple[str | None, dict[str, Any]]],
    ) -> None:
        """Drive a recorded trace through the app. Each entry is
        `(action_name, expected_state_after)`. action=None means
        "check the current state without firing anything" -- the
        conventional first entry for the initial state. Raises
        ReplayDivergence at the first mismatched variable."""
        for step_idx, (action, expected) in enumerate(trace):
            if action is not None:
                self.fire(action)
            actual = self.state_snapshot()
            diffs = {
                key: (exp, actual.get(key))
                for key, exp in expected.items()
                if actual.get(key) != exp
            }
            if diffs:
                raise ReplayDivergence(step_idx, action, diffs)

    def snapshot(self) -> dict:
        """Capture full runtime state (components + channels +
        buffers + RNG) for later `restore()`. Use for counterfactual
        branching."""
        return {
            "components": {
                alias: (
                    comp.state,
                    {n: getattr(comp, n) for n in comp._typed_var_defaults},
                    # Backlog 242: the extra FSM states a multi-inline composite
                    # carries. CRITICAL for the cross-target BFS -- the frontier
                    # is saved/restored through here, so two distinct multi-FSM
                    # states must round-trip distinctly (else the walk explores
                    # from a collapsed state and silently misses reachable ones).
                    # Empty for every single-FSM component.
                    {n: getattr(comp, n) for n in comp._state_fields},
                )
                for alias, comp in self._components.items()
            },
            "channels": [ch.snapshot() for ch in self._channels],
            "buffers": [buf.snapshot() for buf in self._buffers],
            "rng_state": self._rng.getstate(),
            "last_action": self._last_action,
        }

    def restore(self, snap: dict) -> None:
        for alias, entry in snap["components"].items():
            comp = self._components[alias]
            state, typed_vars, state_fields = entry
            comp.state = state
            for name, val in typed_vars.items():
                setattr(comp, name, val)
            # Backlog 242: restore the extra multi-inline FSM states.
            for name, val in state_fields.items():
                setattr(comp, name, val)
        for ch, ch_snap in zip(self._channels, snap["channels"]):
            ch.restore(ch_snap)
        for buf, buf_snap in zip(self._buffers, snap.get("buffers", ())):
            buf.restore(buf_snap)
        self._rng.setstate(snap["rng_state"])
        self._last_action = snap["last_action"]


# ---------------------------------------------------------------------------
# Binding-replay helpers
# ---------------------------------------------------------------------------


def _resolve_endpoint(
    composite: Component, endpoint: tuple[str, str],
) -> Component:
    """Map a buffer-binding endpoint `(alias, port)` to the Component
    holding that port. Alias `""` means the composite itself;
    otherwise it's one of the composite's sub-components. The port
    name goes back to the caller untouched -- the runtime looks up
    the Channel on `_port_channels` lazily at fire time.
    """
    alias, _port = endpoint
    if alias == "":
        return composite
    return composite._sub_components[alias]
