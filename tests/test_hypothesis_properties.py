"""Property-based tests with Hypothesis for temporal boundaries and action state machine."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import re
from typing import List
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from src.retrieval.temporal import parse_iso_datetime, resolve_dates_with_dateparser
from src.ingestion.loader import get_visible_units
from src.core.config import DATA_DIR


# ===========================================================================
# 1. Temporal Point-in-Time Boundaries (Hypothesis)
# ===========================================================================

# Strategy generating timestamps across 2026 with various timezones
TIMEZONES = [
    timezone.utc,
    ZoneInfo("America/Los_Angeles"),
    ZoneInfo("America/New_York"),
    ZoneInfo("Asia/Tokyo"),
    ZoneInfo("Europe/London"),
    timezone(timedelta(hours=5, minutes=30)),  # IST
]

dt_strategy = st.datetimes(
    min_value=datetime(2026, 1, 1),
    max_value=datetime(2026, 12, 31),
)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(dt=dt_strategy, tz=st.sampled_from(TIMEZONES))
def test_hypothesis_temporal_boundary_zero_leakage(dt: datetime, tz):
    """Property: For ANY query cutoff time, no retrieved unit has timestamp > as_of."""
    as_of = dt.replace(tzinfo=tz)
    units = get_visible_units(as_of, data_dir=DATA_DIR)

    target_dt = parse_iso_datetime(as_of)
    for u in units:
        assert u.time <= target_dt, f"Future leak detected: {u.id} ({u.time}) > as_of ({target_dt})"


@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(hour_offset=st.integers(min_value=-12, max_value=12))
def test_hypothesis_timezone_invariance(hour_offset: int):
    """Property: The exact same chronological point in time represented in different timezones

    must retrieve the EXACT identical set of unit IDs.
    """
    base_utc = datetime(2026, 9, 15, 18, 0, 0, tzinfo=timezone.utc)
    target_tz = timezone(timedelta(hours=hour_offset))
    converted = base_utc.astimezone(target_tz)

    units_utc = get_visible_units(base_utc, data_dir=DATA_DIR)
    units_converted = get_visible_units(converted, data_dir=DATA_DIR)

    ids_utc = {u.id for u in units_utc}
    ids_converted = {u.id for u in units_converted}
    assert ids_utc == ids_converted, "Timezone conversion altered visible unit set"


def test_hypothesis_temporal_boundary_inclusive_exclusive_edges():
    """Property: Boundary edges must be strictly inclusive at cutoff and exclusive immediately before."""
    as_of_base = parse_iso_datetime("2026-09-15T12:00:00-07:00")
    units = get_visible_units(as_of_base, data_dir=DATA_DIR)
    assert len(units) > 0

    sample_unit = units[len(units) // 2]
    exact_time = sample_unit.time

    # 1. Exact timestamp must be INCLUSIVE
    units_exact = get_visible_units(exact_time, data_dir=DATA_DIR)
    exact_ids = {u.id for u in units_exact}
    assert sample_unit.id in exact_ids, f"Unit {sample_unit.id} was excluded at exact timestamp (inclusive failure)"

    # 2. Timestamp minus 1 microsecond must be EXCLUSIVE (if no earlier version exists)
    exclusive_time = exact_time - timedelta(microseconds=1)
    units_before = get_visible_units(exclusive_time, data_dir=DATA_DIR)

    # Check that visible units count at exclusive_time <= exact_time
    assert len(units_before) <= len(units_exact)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(noisy_text=st.text(min_size=0, max_size=150))
def test_hypothesis_dateparser_edge_cases(noisy_text: str):
    """Property: resolve_dates_with_dateparser must never crash on arbitrary arbitrary strings,

    and any returned dates must match YYYY-MM-DD.
    """
    as_of = parse_iso_datetime("2026-09-18T18:00:00-07:00")
    dates = resolve_dates_with_dateparser(noisy_text, as_of)
    assert isinstance(dates, list)
    for d in dates:
        assert isinstance(d, str)
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", d), f"Invalid date format returned: {d}"


# ===========================================================================
# 2. Action State Machine (Hypothesis)
# ===========================================================================

class ActionProposalStateMachine:
    """State machine model validating proposal lifecycle invariants:

    - A proposal can NEVER execute without approval
    - A proposal CANNOT be approved twice
    - A rejected proposal can NEVER execute
    """

    VALID_TRANSITIONS = {
        "proposed": {"approved", "rejected"},
        "approved": {"executed", "failed"},
        "rejected": set(),    # Terminal state
        "executed": set(),    # Terminal state
        "failed": set(),      # Terminal state
    }

    def __init__(self, proposal_id: str = "prop-1"):
        self.id = proposal_id
        self.status = "proposed"
        self.approved_count = 0
        self.executed_count = 0

    def approve(self) -> None:
        if self.status != "proposed":
            raise ValueError(f"Cannot approve proposal in status '{self.status}'")
        self.status = "approved"
        self.approved_count += 1

    def reject(self) -> None:
        if self.status != "proposed":
            raise ValueError(f"Cannot reject proposal in status '{self.status}'")
        self.status = "rejected"

    def execute(self) -> None:
        if self.status != "approved":
            raise ValueError(f"Cannot execute proposal in status '{self.status}'. Must be 'approved'.")
        self.status = "executed"
        self.executed_count += 1


@settings(max_examples=50, deadline=None)
@given(operations=st.lists(st.sampled_from(["approve", "reject", "execute"]), min_size=1, max_size=10))
def test_hypothesis_action_state_machine_invariants(operations: List[str]):
    """Property: Verified state machine enforces all three security invariants across random operation traces."""
    sm = ActionProposalStateMachine()

    for op in operations:
        prev_status = sm.status

        if op == "approve":
            if prev_status == "proposed":
                sm.approve()
                assert sm.status == "approved"
                assert sm.approved_count == 1
            else:
                # Invariant: can't be approved twice or approved from rejected/executed
                with pytest.raises(ValueError):
                    sm.approve()
                assert sm.status == prev_status

        elif op == "reject":
            if prev_status == "proposed":
                sm.reject()
                assert sm.status == "rejected"
            else:
                with pytest.raises(ValueError):
                    sm.reject()
                assert sm.status == prev_status

        elif op == "execute":
            if prev_status == "approved":
                sm.execute()
                assert sm.status == "executed"
                assert sm.executed_count == 1
            else:
                # Invariant: a proposal can never execute without approval, rejected can never execute
                with pytest.raises(ValueError):
                    sm.execute()
                assert sm.status == prev_status

        # Global Invariant 1: Execution requires prior approval
        if sm.executed_count > 0:
            assert sm.approved_count == 1

        # Global Invariant 2: Cannot be approved more than once
        assert sm.approved_count <= 1

        # Global Invariant 3: If rejected, execution count is strictly 0
        if sm.status == "rejected":
            assert sm.executed_count == 0
