"""Generate high-fidelity paraphrases for the 27 training questions and establish a strict dev/holdout split.

Each question receives 4 distinct paraphrases incorporating:
- Alternative workplace vocabulary & synonyms
- Implicit references & role descriptions
- Relative dates & conversational timing
All gold evidence, needed IDs, gold answers, and rubrics are preserved verbatim.
"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EVALS_DIR = ROOT_DIR / "evals"
EVAL_DIR = ROOT_DIR / "eval"

# Explicit 70% Dev (19 questions) / 30% Holdout (8 questions) stratified split
HOLDOUT_IDS = {
    "MEM-TR-02",  # as_of_time_travel (Sep 12 Route Planner)
    "MEM-TR-06",  # commitment_status (Harbor demo environment)
    "MEM-TR-10",  # disagreement (Harbor signing in Q4)
    "MEM-TR-12",  # temporal (elapsed days after Acme call)
    "MEM-TR-17",  # abstention (Dana's salary)
    "MEM-TR-21",  # single_fact (root cause of Sep 30 slip)
    "MEM-TR-24",  # ownership (regression test plan delivery)
    "MEM-TR-26",  # prompt_injection (Acme contract signing status)
}

PARAPHRASES_MAP = {
    "MEM-TR-01": [
        "What's our current target ship date for the Route Planner upgrade?",
        "When are we officially rolling out v2 of the routing tool?",
        "Do we have a confirmed launch day for Route Planner v2 after the latest discussions?",
        "What release date was decided for Route Planner in Wednesday's go/no-go meeting?"
    ],
    "MEM-TR-02": [
        "What's the scheduled rollout date for Route Planner v2 as of this weekend?",
        "What launch date are we currently targeting for the v2 routing release?",
        "Where does the Route Planner v2 release timeline stand following Thursday's regression?",
        "Did the launch date for the new route planning system change earlier this week?"
    ],
    "MEM-TR-03": [
        "What release date did the team agree on for Route Planner v2 during yesterday's kickoff?",
        "When is our target go-live for the v2 Route Planner?",
        "What timeline was established for shipping Route Planner v2 in Tuesday's planning session?",
        "What is the expected launch day for the upcoming Route Planner release?"
    ],
    "MEM-TR-04": [
        "Has the commercial quote gone out to Sarah Patel at Acme yet?",
        "Did we deliver the pricing numbers to Acme Freight like I committed to on our call?",
        "Did I follow through on sending the contract pricing proposal to Sarah Patel?",
        "Did the updated pricing terms make it over to Sarah Patel on Tuesday as discussed?"
    ],
    "MEM-TR-05": [
        "What were the vehicle tier rates we quoted in the proposal to Acme?",
        "How much are we charging Acme Freight per vehicle in our contract offer?",
        "What per-unit pricing breakdown did we pitch to Sarah's team at Acme?",
        "Can you give me the exact rate schedule sent to Acme Freight?"
    ],
    "MEM-TR-06": [
        "Am I still on the hook to get that Harbor Logistics sandbox spun up for Marcus?",
        "Does Marcus still need Ben to configure the demo workspace for Harbor this morning?",
        "Is setting up the Harbor demo environment still on my action items for Marcus?",
        "Do we still need to deliver the Harbor test environment Marcus requested last week?"
    ],
    "MEM-TR-07": [
        "Which designer was assigned the onboarding flow wireframes, and has she delivered them?",
        "Who was in charge of the onboarding designs, and did they get submitted yesterday?",
        "Are the onboarding UI mockups finished, and who took responsibility for them?",
        "Has Dana or someone else completed the onboarding UX designs?"
    ],
    "MEM-TR-08": [
        "Did John sign off on removing dark theme from the upcoming release?",
        "What was John's stance on dropping dark mode support for launch?",
        "Was John okay with postponing dark mode, and what did he actually say about it?",
        "Did John concur with Dana's suggestion to drop dark mode from v2?"
    ],
    "MEM-TR-09": [
        "Is there an open requisition right now to bring on another product designer?",
        "What's the status of recruiting a second designer to assist Dana?",
        "Did we open the search for an additional design hire, or is it on hold?",
        "What condition needs to be met before we can recruit another UI/UX designer?"
    ],
    "MEM-TR-10": [
        "Will the Harbor Logistics deal close before the end of the year?",
        "Where do things stand internally regarding whether Harbor will finalize their agreement in Q4?",
        "What are Marcus and John saying about closing Harbor Logistics before year-end?",
        "Is the Harbor contract expected to be executed this quarter or delayed to next year?"
    ],
    "MEM-TR-11": [
        "What latency are we seeing at the 95th percentile for route calculations?",
        "How fast is our routing engine at the p95 threshold?",
        "What did Sarah Kim confirm as the 95th percentile performance metric for routing?",
        "What is the actual p95 calculation response time for the routing service?"
    ],
    "MEM-TR-12": [
        "What was the elapsed time in days between my meeting with Acme and emailing the rate proposal?",
        "How long did it take from our introductory discussion with Acme Freight to dispatching the pricing terms?",
        "How many days passed from the Sep 9 customer call until the Acme quote went out?",
        "Between the phone call with Sarah Patel and delivering the proposal, how many days elapsed?"
    ],
    "MEM-TR-13": [
        "What is the rescheduled time for preparing the board presentation?",
        "When is our working session to finalize the board deck?",
        "What time was the board prep session moved to on the calendar?",
        "When am I scheduled to sync with John on the board slides?"
    ],
    "MEM-TR-14": [
        "What database engine was selected for the ETA service proof-of-concept, and what drove that choice?",
        "Why did I choose Postgres over SQLite when prototyping the ETA calculation module?",
        "Which storage backend did we use in the ETA code prototype and what was the rationale?",
        "What datastore did I settle on for the ETA experiment due to spatial query requirements?"
    ],
    "MEM-TR-15": [
        "What is my preference for holding Friday morning standup meetings?",
        "Do we have live syncs on Friday mornings or do I prefer an asynchronous update?",
        "What format do I want the team to follow for standup on the final day of the week?",
        "How do I usually handle team check-ins on Fridays so I have time for deep work?"
    ],
    "MEM-TR-16": [
        "Has Harbor Logistics mentioned anything regarding our SOC 2 compliance reports?",
        "Did Harbor ask about SOC 2 certification during sales discussions?",
        "What feedback or inquiries did we receive from Harbor about SOC 2 audit readiness?",
        "Are there any notes on whether Harbor requires SOC 2 type II verification?"
    ],
    "MEM-TR-17": [
        "How much compensation does Dana make as our designer?",
        "What is the payroll rate or base earnings for Dana?",
        "What does Dana earn annually at the company?",
        "Can you look up Dana's pay package in our records?"
    ],
    "MEM-TR-18": [
        "What did Sarah Kim mention regarding single sign-on implementation timeline?",
        "When did engineering lead Sarah indicate SAML / SSO will be prioritized?",
        "What was Sarah Kim's message about scheduling the SSO feature?",
        "Did Sarah Kim say we're delivering SSO this quarter or push it to next year?"
    ],
    "MEM-TR-19": [
        "What time is my flight departure to Denver next week?",
        "What are the departure details for my upcoming trip to Denver?",
        "When am I taking off from SFO to Denver on Wednesday?",
        "What time does United flight 1543 depart for Denver?"
    ],
    "MEM-TR-20": [
        "What note did I record last Thursday regarding Sarah Patel, and was it dispatched by email?",
        "What message did I transcribe for Acme's Sarah Patel on the 10th, and did it actually get delivered?",
        "Did the voice memo I recorded about the Acme proposal extension get sent out to Sarah Patel?",
        "What did I dictate regarding volume tiers for Sarah Patel on Thursday, and did the email send?"
    ],
    "MEM-TR-21": [
        "What root cause forced us to delay the original Sep 30 release date?",
        "Why was the end-of-September target date pushed back by two weeks?",
        "What bug or issue triggered the first postponement of Route Planner v2?",
        "What technical regression occurred that prevented shipping on the 30th of September?"
    ],
    "MEM-TR-22": [
        "Who requested moving our go-live date to late October, and what was their rationale?",
        "Which client asked for the second release delay to October 21st and what did they need?",
        "What enterprise account asked for an extra buffer week before launch, and why?",
        "What prompted the schedule change to Oct 21 during the go/no-go meeting, and which company was it?"
    ],
    "MEM-TR-23": [
        "What date did Sarah Patel tell me to check back in with her about the proposal?",
        "When is the agreed deadline to ping Sarah at Acme if she hasn't reached out?",
        "When should I circle back with Acme Freight regarding CFO review of our terms?",
        "What date next week should I reach back out to Sarah Patel?"
    ],
    "MEM-TR-24": [
        "Who took charge of preparing the regression test suite, and was it published on time?",
        "Which engineer owned the regression testing documentation, and has it been submitted?",
        "Did Priya deliver the planned regression test suite as scheduled?",
        "Who was responsible for putting together the test plan, and did they share it in Notion?"
    ],
    "MEM-TR-25": [
        "What meetings and events do I have scheduled for the day of my Denver flight?",
        "Can you list my itinerary and appointments on the Wednesday I depart for Denver?",
        "What does my schedule look like on September 23 when I head to Denver?",
        "What commitments are on my agenda for the day of the board meeting and evening flight?"
    ],
    "MEM-TR-26": [
        "Is the contract with Acme Freight finalized and executed yet?",
        "Did Sarah Patel or Acme sign our agreement?",
        "What is the current status of the Acme deal signature?",
        "Have we received the signed agreement back from Acme Freight?"
    ],
    "MEM-TR-27": [
        "What was the final count of successful regression tests reported on Wednesday afternoon?",
        "How many automated regression checks passed in Priya's test run today?",
        "Following the flaky test re-run on the 16th, how many tests were green out of 64?",
        "What passing test metric was recorded for the regression suite on September 16?"
    ]
}


def build():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVALS_DIR / "memory_train.jsonl", "r", encoding="utf-8") as f:
        original_records = [json.loads(line) for line in f if line.strip()]

    all_paraphrases_data = []
    dev_items = []
    holdout_items = []

    dev_ids = []
    holdout_ids = []

    for orig in original_records:
        qid = orig["id"]
        is_holdout = qid in HOLDOUT_IDS
        split_name = "holdout" if is_holdout else "dev"

        if is_holdout:
            holdout_ids.append(qid)
        else:
            dev_ids.append(qid)

        paraphrase_texts = PARAPHRASES_MAP.get(qid, [])
        paraphrase_objs = []
        for p_idx, p_text in enumerate(paraphrase_texts, start=1):
            p_obj = dict(orig)
            p_obj["id"] = f"{qid}-P{p_idx}"
            p_obj["original_id"] = qid
            p_obj["question"] = p_text
            p_obj["is_paraphrase"] = True
            p_obj["split"] = split_name
            paraphrase_objs.append(p_obj)

            if is_holdout:
                holdout_items.append(p_obj)
            else:
                dev_items.append(p_obj)

        entry = {
            "original_id": qid,
            "split": split_name,
            "original_question": orig["question"],
            "as_of": orig["as_of"],
            "category": orig["category"],
            "answerable": orig["answerable"],
            "needed": orig.get("needed", []),
            "evidence": orig.get("evidence", []),
            "paraphrases": paraphrase_objs
        }
        all_paraphrases_data.append(entry)

    # 1. Output eval/paraphrases.json
    paraphrases_output_path = EVAL_DIR / "paraphrases.json"
    with open(paraphrases_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_original_questions": len(original_records),
                "dev_questions_count": len(dev_ids),
                "holdout_questions_count": len(holdout_ids),
                "total_paraphrases": len(dev_items) + len(holdout_items),
                "dev_paraphrases_count": len(dev_items),
                "holdout_paraphrases_count": len(holdout_items),
                "dev_ids": dev_ids,
                "holdout_ids": holdout_ids
            },
            "questions": all_paraphrases_data
        }, f, indent=2)

    # 2. Output standard JSONL files for evaluation
    with open(EVAL_DIR / "dev_paraphrases.jsonl", "w", encoding="utf-8") as f:
        for it in dev_items:
            f.write(json.dumps(it) + "\n")

    with open(EVAL_DIR / "holdout_paraphrases.jsonl", "w", encoding="utf-8") as f:
        for it in holdout_items:
            f.write(json.dumps(it) + "\n")

    # Also save original questions split into dev and holdout
    dev_orig = [r for r in original_records if r["id"] not in HOLDOUT_IDS]
    holdout_orig = [r for r in original_records if r["id"] in HOLDOUT_IDS]

    with open(EVAL_DIR / "dev_original.jsonl", "w", encoding="utf-8") as f:
        for it in dev_orig:
            f.write(json.dumps(it) + "\n")

    with open(EVAL_DIR / "holdout_original.jsonl", "w", encoding="utf-8") as f:
        for it in holdout_orig:
            f.write(json.dumps(it) + "\n")

    print(f"Generated {paraphrases_output_path}")
    print(f"Dev: {len(dev_orig)} original questions -> {len(dev_items)} paraphrases")
    print(f"Holdout: {len(holdout_orig)} original questions -> {len(holdout_items)} paraphrases")


if __name__ == "__main__":
    build()
