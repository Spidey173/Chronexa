"""API tests with FastAPI TestClient covering:
- Auth failures: no token, bad token, wrong role
- Complete workflow: dry-run propose -> operator approve -> background execution
- State machine edge cases: double approval, approval of rejected, execution of rejected
- Assertion on state changes (DB rows in action_proposals, AuditLog entries with user_id)
"""
import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.deps import EngineState
from src.db.session import SessionLocal
from src.db.models import User, ActionProposal, AuditLog
from src.auth import create_access_token, get_password_hash


@pytest.fixture(scope="module")
def client():
    state = EngineState.get_instance(pre_warm=False)
    if state.action_parser is not None:
        state.action_parser.provider = "none"
    if state.generator is not None:
        state.generator.provider = "none"
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def users(db_session):
    """Seed test users with viewer, operator, and admin roles."""
    roles = ["viewer", "operator", "admin"]
    user_dict = {}
    for r in roles:
        email = f"{r}_{uuid.uuid4().hex[:6]}@chronexa.test"
        u = User(
            id=str(uuid.uuid4()),
            email=email,
            hashed_password=get_password_hash("password123"),
            role=r,
            oauth_tokens={"slack_token": "xoxb-test-token"}
        )
        db_session.add(u)
        user_dict[r] = u
    db_session.commit()
    for r in roles:
        db_session.refresh(user_dict[r])
    return user_dict


def test_auth_failures_no_token(client):
    # 1. Propose without token -> 401
    res = client.post("/actions/propose", json={"command": "Open Figma"})
    assert res.status_code == 401
    assert "missing" in res.json()["detail"].lower() or "bearer" in res.json()["detail"].lower()

    # 2. Approve without token -> 401
    res = client.post("/actions/any-id/approve")
    assert res.status_code == 401

    # 3. Reject without token -> 401
    res = client.post("/actions/any-id/reject")
    assert res.status_code == 401


def test_auth_failures_bad_token(client):
    bad_headers = {"Authorization": "Bearer not.a.valid.jwt.token"}

    res1 = client.post("/actions/propose", json={"command": "Open Figma"}, headers=bad_headers)
    assert res1.status_code == 401
    assert "invalid" in res1.json()["detail"].lower()

    res2 = client.post("/actions/some-id/approve", headers=bad_headers)
    assert res2.status_code == 401


def test_auth_failures_wrong_role_viewer_cannot_approve(client, users):
    viewer = users["viewer"]
    viewer_token = create_access_token({"sub": viewer.id, "email": viewer.email, "role": viewer.role})
    viewer_headers = {"Authorization": f"Bearer {viewer_token}"}

    # Viewer can propose
    res_prop = client.post("/actions/propose", json={"command": "Open Figma"}, headers=viewer_headers)
    assert res_prop.status_code == 200
    prop_id = res_prop.json()["id"]

    # Viewer CANNOT approve -> 403 Forbidden
    res_appr = client.post(f"/actions/{prop_id}/approve", headers=viewer_headers)
    assert res_appr.status_code == 403
    assert "operator" in res_appr.json()["detail"].lower()


def test_full_propose_approve_execute_flow_with_state_assertions(client, users, db_session):
    operator = users["operator"]
    op_token = create_access_token({"sub": operator.id, "email": operator.email, "role": operator.role})
    op_headers = {"Authorization": f"Bearer {op_token}"}

    # 1. Propose Action
    command_text = "Message Sarah on Slack saying the geocoding fix is confirmed"
    res_prop = client.post("/actions/propose", json={"command": command_text}, headers=op_headers)
    assert res_prop.status_code == 200
    prop_data = res_prop.json()
    prop_id = prop_data["id"]
    assert prop_data["status"] == "proposed"
    assert prop_data["user_id"] == operator.id

    # Assert state in DB immediately after propose
    db_prop = db_session.query(ActionProposal).filter(ActionProposal.id == prop_id).first()
    assert db_prop is not None
    assert db_prop.status == "proposed"
    assert db_prop.user_id == operator.id
    assert db_prop.command == command_text

    # Assert propose written to audit log with operator user_id
    audit_propose = db_session.query(AuditLog).filter(
        AuditLog.resource_id == prop_id,
        AuditLog.action == "propose"
    ).first()
    assert audit_propose is not None
    assert audit_propose.user_id == operator.id

    # 2. Operator Approves Action
    res_appr = client.post(f"/actions/{prop_id}/approve", headers=op_headers)
    assert res_appr.status_code == 200
    assert res_appr.json()["status"] == "approved"

    # Refresh DB session to observe background execution updates
    db_session.expire_all()
    db_prop_after = db_session.query(ActionProposal).filter(ActionProposal.id == prop_id).first()
    assert db_prop_after.status == "executed"
    assert db_prop_after.execution_result is not None
    assert len(db_prop_after.execution_result) > 0

    # Assert approve and execute entries written to audit log
    audit_approve = db_session.query(AuditLog).filter(
        AuditLog.resource_id == prop_id,
        AuditLog.action == "approve"
    ).first()
    assert audit_approve is not None
    assert audit_approve.user_id == operator.id

    audit_execute = db_session.query(AuditLog).filter(
        AuditLog.resource_id == prop_id,
        AuditLog.action == "execute"
    ).first()
    assert audit_execute is not None
    assert audit_execute.user_id == operator.id


def test_state_machine_double_approval_rejected(client, users, db_session):
    operator = users["operator"]
    op_token = create_access_token({"sub": operator.id, "email": operator.email, "role": operator.role})
    op_headers = {"Authorization": f"Bearer {op_token}"}

    # Propose
    res_prop = client.post("/actions/propose", json={"command": "Open Figma"}, headers=op_headers)
    prop_id = res_prop.json()["id"]

    # First approval -> 200
    res_appr1 = client.post(f"/actions/{prop_id}/approve", headers=op_headers)
    assert res_appr1.status_code == 200

    # Second approval -> 400 (Cannot be approved twice)
    res_appr2 = client.post(f"/actions/{prop_id}/approve", headers=op_headers)
    assert res_appr2.status_code == 400
    assert "already" in res_appr2.json()["detail"].lower()


def test_state_machine_rejected_cannot_be_approved_or_executed(client, users, db_session):
    operator = users["operator"]
    op_token = create_access_token({"sub": operator.id, "email": operator.email, "role": operator.role})
    op_headers = {"Authorization": f"Bearer {op_token}"}

    # Propose
    res_prop = client.post("/actions/propose", json={"command": "Open Terminal"}, headers=op_headers)
    prop_id = res_prop.json()["id"]

    # Reject -> 200
    res_rej = client.post(f"/actions/{prop_id}/reject", headers=op_headers)
    assert res_rej.status_code == 200
    assert res_rej.json()["status"] == "rejected"

    # Assert DB status is rejected
    db_session.expire_all()
    p = db_session.query(ActionProposal).filter(ActionProposal.id == prop_id).first()
    assert p.status == "rejected"

    # Assert reject logged in audit_log
    audit_rej = db_session.query(AuditLog).filter(
        AuditLog.resource_id == prop_id,
        AuditLog.action == "reject"
    ).first()
    assert audit_rej is not None
    assert audit_rej.user_id == operator.id

    # Attempting to approve a rejected proposal MUST fail with 400
    res_appr = client.post(f"/actions/{prop_id}/approve", headers=op_headers)
    assert res_appr.status_code == 400
    assert "rejected" in res_appr.json()["detail"].lower()

    # Proposal remains in rejected state, never executed
    db_session.expire_all()
    p_final = db_session.query(ActionProposal).filter(ActionProposal.id == prop_id).first()
    assert p_final.status == "rejected"
    assert p_final.execution_result is None
