"""Contextual Chunker and Passage Windowing Engine.

Enriches raw records with conversational sliding windows and structured metadata:
- Meetings: windowed by speaker turn (t-1, t, t+1) with timestamps and speaker labels.
- Slack: windowed with immediate conversational context in the channel/thread.
- Email: structured with participants, subject, body, and dates.
- Calendar: structured with event window, attendees, and status.
- Strict ID preservation: each chunk preserves the canonical unit ID expected by the evaluation harness.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from eval_harness.records import Unit
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


def _dt(s) -> datetime:
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d


@dataclass
class EnrichedUnit(Unit):
    source_type: str = "unknown"
    speaker: str = ""
    parent_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_enriched_records(data_dir: Path) -> Tuple[List[EnrichedUnit], Dict[str, datetime], Dict[str, list]]:
    d = Path(data_dir)
    units: List[EnrichedUnit] = []
    deleted: Dict[str, datetime] = {}
    edits: Dict[str, list] = {}

    # 1. Meetings: Sliding window over speaker turns (t-1, t, t+1)
    for f in sorted((d / "native/meetings").glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        start = _dt(m["start"])
        segs = m.get("segments", [])
        m_title = m.get("title", "Meeting")
        m_date = m["start"][:10]

        for i, s in enumerate(segs):
            who = s.get("speaker_name") or s.get("speaker_label") or "Unknown speaker"
            curr_text = s.get("text", "")
            seg_time = start + timedelta(seconds=s.get("end_s", 0))

            # Build conversational context window
            context_parts = []
            if i > 0:
                prev_s = segs[i - 1]
                prev_who = prev_s.get("speaker_name") or prev_s.get("speaker_label") or "Speaker"
                context_parts.append(f"[{prev_who}]: {prev_s.get('text', '')}")

            context_parts.append(f">> [{who}]: {curr_text}")

            if i < len(segs) - 1:
                next_s = segs[i + 1]
                next_who = next_s.get("speaker_name") or next_s.get("speaker_label") or "Speaker"
                context_parts.append(f"[{next_who}]: {next_s.get('text', '')}")

            window_body = " ".join(context_parts)
            header = f"[Meeting: {m_title} | Date: {m_date} | Speaker: {who} | ID: {s['seg_id']}]"
            enriched_text = f"{header}\n{window_body}"

            unit = EnrichedUnit(
                id=s["seg_id"],
                record=m["id"],
                time=seg_time,
                text=enriched_text,
                source_type="meeting",
                speaker=who,
                parent_title=m_title,
                metadata={"title": m_title, "meeting_id": m["id"], "start": m["start"]}
            )
            units.append(unit)

    # 2. Slack: Message context with previous turn
    users_file = d / "connectors/slack/users.json"
    names = {u["id"]: u["real_name"] for u in json.loads(users_file.read_text(encoding="utf-8"))} if users_file.exists() else {}
    chans_file = d / "connectors/slack/channels.json"
    chans = {c["id"]: c["name"] for c in json.loads(chans_file.read_text(encoding="utf-8"))} if chans_file.exists() else {}

    slack_msgs = []
    slack_file = d / "connectors/slack/messages.jsonl"
    if slack_file.exists():
        with open(slack_file, encoding="utf-8") as sf:
            for line in sf:
                if line.strip():
                    slack_msgs.append(json.loads(line))

    # Sort slack messages by time for windowing
    slack_msgs.sort(key=lambda x: _dt(x["ts"]))
    for i, x in enumerate(slack_msgs):
        t = _dt(x["ts"])
        where = chans.get(x["channel_id"], x.get("channel_id", "general"))
        if x.get("subtype") == "message_deleted":
            deleted[x["target_id"]] = t
            units.append(EnrichedUnit(
                x["id"], x["id"], t,
                f"[Slack #{where}] (message {x['target_id']} was deleted)",
                source_type="slack"
            ))
            continue
        if x.get("subtype") == "message_changed":
            edits.setdefault(x["target_id"], []).append((t, x["text"]))
            units.append(EnrichedUnit(
                x["id"], x["id"], t,
                f"[Slack #{where}, edit of {x['target_id']}] {x['text']}",
                source_type="slack"
            ))
            continue

        who = names.get(x.get("user"), x.get("bot_name") or x.get("user", "Unknown"))
        text = x.get("text", "")
        
        # Look back 1 message if within same channel and within 2 hours
        prev_ctx = ""
        if i > 0:
            prev_x = slack_msgs[i - 1]
            if prev_x.get("channel_id") == x.get("channel_id") and (t - _dt(prev_x["ts"])) < timedelta(hours=2):
                prev_who = names.get(prev_x.get("user"), prev_x.get("user", "User"))
                prev_ctx = f"Context: [{prev_who}]: {prev_x.get('text', '')}\n"

        header = f"[Slack #{where} | Date: {x['ts'][:10]} | User: {who} | ID: {x['id']}]"
        enriched_text = f"{header}\n{prev_ctx}{who}: {text}"

        units.append(EnrichedUnit(
            id=x["id"],
            record=x["id"],
            time=t,
            text=enriched_text,
            source_type="slack",
            speaker=who,
            parent_title=f"#{where}"
        ))

    # 3. Gmail: Structured headers + message body
    gmail_file = d / "connectors/gmail/messages.jsonl"
    if gmail_file.exists():
        with open(gmail_file, encoding="utf-8") as gf:
            for line in gf:
                if not line.strip():
                    continue
                x = json.loads(line)
                t = _dt(x["date"])
                rec_id = x["id"]
                to_str = ", ".join(x.get("to", []))
                cc_str = f" Cc: {', '.join(x.get('cc', []))}" if x.get("cc") else ""
                subj = x.get("subject", "No subject")
                header = f"[Email | Date: {x['date'][:10]} | Subject: {subj} | From: {x.get('from')} | To: {to_str}{cc_str} | ID: {rec_id}]"
                enriched_text = f"{header}\n{x.get('body', '')}"

                units.append(EnrichedUnit(
                    id=rec_id,
                    record=rec_id,
                    time=t,
                    text=enriched_text,
                    source_type="email",
                    speaker=x.get("from", ""),
                    parent_title=subj
                ))

    # 4. Google Calendar: Structured events
    cal_file = d / "connectors/google_calendar/events.jsonl"
    if cal_file.exists():
        with open(cal_file, encoding="utf-8") as cf:
            for line in cf:
                if not line.strip():
                    continue
                x = json.loads(line)
                st, en = x["start"], x["end"]
                start_iso = st.get("dateTime") or st.get("date")
                end_iso = en.get("dateTime") or en.get("date")
                when = f"{start_iso} to {end_iso}"
                att = ", ".join(a.get("email", "") for a in x.get("attendees", []))
                loc = f" | Location: {x.get('location')}" if x.get("location") else ""
                header = f"[Calendar | Event: {x.get('summary')} | Time: {when}{loc} | Attendees: {att} | ID: {x['id']}]"
                enriched_text = f"{header}\n{x.get('description', '')}"

                units.append(EnrichedUnit(
                    id=x["id"],
                    record=x["id"],
                    time=_dt(x["updated"]),
                    text=enriched_text,
                    source_type="calendar",
                    speaker="Organizer",
                    parent_title=x.get("summary", "")
                ))

    # 5. Dictations
    dct_file = d / "native/dictation/dictations.jsonl"
    if dct_file.exists():
        with open(dct_file, encoding="utf-8") as df:
            for line in df:
                if not line.strip():
                    continue
                x = json.loads(line)
                t = _dt(x["timestamp"])
                header = f"[Dictation | Target: {x.get('target_app')} – {x.get('target_context')} | Date: {x['timestamp'][:10]} | ID: {x['id']}]"
                enriched_text = f"{header}\n{x.get('cleaned_text', '')}"
                if x.get("raw_transcript"):
                    enriched_text += f"\n(raw transcript: {x['raw_transcript']})"

                units.append(EnrichedUnit(
                    id=x["id"],
                    record=x["id"],
                    time=t,
                    text=enriched_text,
                    source_type="dictation",
                    speaker="Alex Rivera",
                    parent_title=x.get("target_app", "")
                ))

    # 6. ChatGPT Conversations
    cgpt_file = d / "connectors/chatgpt/conversations.json"
    if cgpt_file.exists():
        for c in json.loads(cgpt_file.read_text(encoding="utf-8")):
            title = c.get("title", "ChatGPT")
            for m in c.get("messages", []):
                t = _dt(m["create_time"])
                header = f"[ChatGPT | Topic: {title} | Role: {m.get('role')} | Date: {m['create_time'][:10]} | ID: {m['id']}]"
                enriched_text = f"{header}\n{m.get('content', '')}"
                units.append(EnrichedUnit(
                    id=m["id"],
                    record=c["id"],
                    time=t,
                    text=enriched_text,
                    source_type="chatgpt",
                    speaker=m.get("role", ""),
                    parent_title=title
                ))

    # 7. Codex Sessions
    for f in sorted((d / "connectors/codex/sessions").glob("*.jsonl")):
        with open(f, encoding="utf-8") as cf:
            events = [json.loads(l) for l in cf if l.strip()]
        if not events:
            continue
        meta, body = events[0], events[1:]
        text_body = "\n".join(f"{e.get('role', e.get('tool', e.get('type')))}: {e.get('content') or e.get('input', '')}" for e in body)
        t = _dt(body[-1]["timestamp"] if body else meta["started_at"])
        header = f"[Codex Session | Repo: {meta.get('repo')} | ID: {meta['id']}]"
        enriched_text = f"{header}\n{text_body}"
        units.append(EnrichedUnit(
            id=meta["id"],
            record=meta["id"],
            time=t,
            text=enriched_text,
            source_type="codex",
            speaker="User/Agent",
            parent_title=meta.get("repo", "")
        ))

    units.sort(key=lambda u: u.time)
    return units, deleted, edits
