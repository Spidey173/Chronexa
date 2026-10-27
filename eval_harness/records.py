"""Index of every citable id in data/: when it became available, which record it belongs to,
its text, and edits/deletions. Used by the scorers (time-leak and deletion checks). Standard library only.

A unit is anything a system can cite: a meeting segment, a dictation, a Slack message or edit,
an email, a calendar event, a Codex session, a ChatGPT message. A record is what contains it
(a meeting, a ChatGPT conversation); for most sources unit and record are the same id.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Unit:
    id: str
    record: str
    time: datetime
    text: str
    source_type: str = "unknown"
    speaker: str = ""
    meeting_id: str = ""
    timestamp_iso: str = ""


from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


def _dt(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d


def load(data_dir):
    d = Path(data_dir)
    units = []
    deleted = {}   # target id -> deletion time
    edits = {}     # target id -> [(time, new text)]

    for f in sorted((d / "native/meetings").glob("*.json")):
        m = json.loads(f.read_text())
        start = _dt(m["start"])
        for s in m["segments"]:
            who = s.get("speaker_name") or s.get("speaker_label") or "Unknown speaker"
            seg_time = start + timedelta(seconds=s["end_s"])
            units.append(Unit(s["seg_id"], m["id"], seg_time,
                              f"[{m['title']}, {m['start'][:10]}] {who}: {s['text']}",
                              source_type="meeting", speaker=who, meeting_id=m["id"],
                              timestamp_iso=seg_time.isoformat()))

    dict_file = d / "native/dictation/dictations.jsonl"
    if dict_file.exists():
        with open(dict_file, encoding="utf-8") as df:
            for line in df:
                x = json.loads(line)
                t = _dt(x["timestamp"])
                units.append(Unit(x["id"], x["id"], t,
                                  f"[Dictation {x['mode']} into {x['target_app']} – {x['target_context']}, "
                                  f"{x['delivery_state']}] {x['cleaned_text']}"
                                  f"{chr(10) + '(raw transcript: ' + x['raw_transcript'] + ')' if x.get('raw_transcript') else ''}",
                                  source_type="dictation", speaker="Alex Rivera", meeting_id=x.get("target_app", ""),
                                  timestamp_iso=t.isoformat()))

    slack_users = d / "connectors/slack/users.json"
    names = {u["id"]: u["real_name"] for u in json.loads(slack_users.read_text(encoding="utf-8"))} if slack_users.exists() else {}
    slack_chans = d / "connectors/slack/channels.json"
    chans = {c["id"]: c["name"] for c in json.loads(slack_chans.read_text(encoding="utf-8"))} if slack_chans.exists() else {}

    slack_msgs_file = d / "connectors/slack/messages.jsonl"
    if slack_msgs_file.exists():
        with open(slack_msgs_file, encoding="utf-8") as sf:
            for line in sf:
                x = json.loads(line)
                t = _dt(x["ts"])
                where = chans.get(x["channel_id"], x["channel_id"])
                if x.get("subtype") == "message_deleted":
                    deleted[x["target_id"]] = t
                    units.append(Unit(x["id"], x["id"], t, f"[Slack {where}] (message {x['target_id']} was deleted)",
                                      source_type="slack", speaker="System", meeting_id=where, timestamp_iso=t.isoformat()))
                    continue
                if x.get("subtype") == "message_changed":
                    edits.setdefault(x["target_id"], []).append((t, x["text"]))
                    units.append(Unit(x["id"], x["id"], t, f"[Slack {where}, edit of {x['target_id']}] {x['text']}",
                                      source_type="slack", speaker="System", meeting_id=where, timestamp_iso=t.isoformat()))
                    continue
                who = names.get(x.get("user"), x.get("bot_name") or x.get("user"))
                units.append(Unit(x["id"], x["id"], t, f"[Slack {where}] {who}: {x['text']}",
                                  source_type="slack", speaker=who, meeting_id=where, timestamp_iso=t.isoformat()))

    gmail_file = d / "connectors/gmail/messages.jsonl"
    if gmail_file.exists():
        with open(gmail_file, encoding="utf-8") as gf:
            for line in gf:
                x = json.loads(line)
                t = _dt(x["date"])
                units.append(Unit(x["id"], x["id"], t,
                                  f"[Email {x['date'][:16]}] From {x['from']} To {', '.join(x['to'])}"
                                  f"{' Cc ' + ', '.join(x['cc']) if x['cc'] else ''} | {x['subject']}\n{x['body']}",
                                  source_type="email", speaker=x.get("from", ""), meeting_id=x.get("subject", ""),
                                  timestamp_iso=t.isoformat()))

    cal_file = d / "connectors/google_calendar/events.jsonl"
    if cal_file.exists():
        with open(cal_file, encoding="utf-8") as cf:
            for line in cf:
                x = json.loads(line)
                st, en = x["start"], x["end"]
                when = f"{st.get('dateTime') or st.get('date')} to {en.get('dateTime') or en.get('date')}"
                att = ", ".join(a["email"] for a in x.get("attendees", []))
                t = _dt(x["updated"])
                units.append(Unit(x["id"], x["id"], t,
                                  f"[Calendar, {x['status']}] {x['summary']} | {when} | {x.get('location') or ''} | "
                                  f"attendees: {att} | {x.get('description') or ''}"
                                  f"{' | repeats ' + str(x['recurrence']) if x.get('recurrence') else ''}",
                                  source_type="calendar", speaker="Organizer", meeting_id=x["id"],
                                  timestamp_iso=t.isoformat()))

    for f in sorted((d / "connectors/codex/sessions").glob("*.jsonl")):
        with open(f, encoding="utf-8") as cf:
            events = [json.loads(l) for l in cf if l.strip()]
        meta, body = events[0], events[1:]
        text = "\n".join(f"{e.get('role', e.get('tool', e['type']))}: {e.get('content') or e.get('input', '')}"
                         for e in body)
        t = _dt(body[-1]["timestamp"] if body else meta["started_at"])
        units.append(Unit(meta["id"], meta["id"], t,
                          f"[Codex session, repo {meta.get('repo')}]\n{text}",
                          source_type="codex", speaker="User/Agent", meeting_id=meta.get("repo", ""),
                          timestamp_iso=t.isoformat()))

    cgpt_file = d / "connectors/chatgpt/conversations.json"
    if cgpt_file.exists():
        for c in json.loads(cgpt_file.read_text(encoding="utf-8")):
            for m in c["messages"]:
                t = _dt(m["create_time"])
                units.append(Unit(m["id"], c["id"], t,
                                  f"[ChatGPT '{c['title']}'] {m['role']}: {m['content']}",
                                  source_type="chatgpt", speaker=m.get("role", ""), meeting_id=c.get("title", ""),
                                  timestamp_iso=t.isoformat()))

    units.sort(key=lambda u: u.time)
    return units, deleted, edits


def context(data_dir):
    """What the scorers need: availability time, record of each unit, word counts, deletions."""
    units, deleted, _ = load(data_dir)
    avail = {u.id: u.time for u in units}
    record_of = {u.id: u.record for u in units}
    for u in units:  # a record id (a whole meeting or conversation) exists once its first unit does
        if u.record not in avail or u.time < avail[u.record]:
            avail[u.record] = min(u.time, avail.get(u.record, u.time))
    words = {u.id: len(u.text.split()) for u in units}
    return {"avail": avail, "record_of": record_of, "words": words, "deleted": deleted}


def visible(data_dir, as_of):
    """Units a system could know at `as_of`: delivered by then, not deleted by then, edits applied."""
    units, deleted, edits = load(data_dir)
    as_of_dt = as_of if isinstance(as_of, datetime) else _dt(as_of)
    if as_of_dt.tzinfo is None:
        as_of_dt = as_of_dt.replace(tzinfo=LOCAL_TZ)
    out = []
    for u in units:
        if u.time > as_of_dt or (u.id in deleted and deleted[u.id] <= as_of_dt):
            continue
        newer = [txt for t, txt in edits.get(u.id, []) if t <= as_of_dt]
        out.append(Unit(u.id, u.record, u.time, u.text if not newer else u.text.partition(": ")[0] + ": " + newer[-1] + " (edited)"))
    return out
