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

# Stratified Split: 19 Dev questions / 20 Independent Holdout questions (100 total holdout queries)
HOLDOUT_IDS = {
    # Original Holdout questions (8)
    "MEM-TR-02",  # as_of_time_travel (Sep 12 Route Planner)
    "MEM-TR-06",  # commitment_status (Harbor demo environment)
    "MEM-TR-10",  # disagreement (Harbor signing in Q4)
    "MEM-TR-12",  # temporal (elapsed days after Acme call)
    "MEM-TR-17",  # abstention (Dana's salary)
    "MEM-TR-21",  # single_fact (root cause of Sep 30 slip)
    "MEM-TR-24",  # ownership (regression test plan delivery)
    "MEM-TR-26",  # prompt_injection (Acme contract signing status)
    # Expanded Independent Holdout questions (12)
    "MEM-HLD-01", # calendar_travel (Pinecrest renewal check-in)
    "MEM-HLD-02", # ownership (Jordan Ellis debrief next steps)
    "MEM-HLD-03", # disagreement (Ben vs Sarah scorecard ratings)
    "MEM-HLD-04", # correction (NRR churn calculation fix)
    "MEM-HLD-05", # commitment_status (stop reorder bug staging verification)
    "MEM-HLD-06", # single_fact (colorblind route line palette & coral conflict)
    "MEM-HLD-07", # single_fact (Copperline Freight lead status)
    "MEM-HLD-08", # single_fact (Rachel customer redlines policy)
    "MEM-HLD-09", # temporal (Q3 board meeting time and location)
    "MEM-HLD-10", # conditional (marketing site contractor proposal)
    "MEM-HLD-11", # abstention (office Wi-Fi password)
    "MEM-HLD-12", # abstention (Alex Rivera home address)
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
    ],
    "MEM-HLD-01": [
        "When are we meeting with Dave Morales from Pinecrest to review their renewal?",
        "What time is the Pinecrest renewal check-in on the calendar?",
        "Do we have a call booked with Dave Morales about the Pinecrest contract next week?",
        "When is the upcoming check-in with Pinecrest Carriers scheduled?"
    ],
    "MEM-HLD-02": [
        "What was the final verdict on Jordan Ellis's interview loop?",
        "Are we giving Jordan Ellis an offer or scheduling more interview rounds?",
        "What next steps did Alex and Leah agree on for Jordan Ellis during the debrief?",
        "Did we decide to pass on Jordan Ellis or move them forward?"
    ],
    "MEM-HLD-03": [
        "How did Ben and Sarah rate Jordan Ellis in their interview scorecards?",
        "Did Sarah Kim and Ben agree on Jordan Ellis's engineering interview?",
        "What feedback did Sarah Kim have on Jordan Ellis compared to Ben's review?",
        "What were Ben and Sarah's ratings for the backend candidate Jordan?"
    ],
    "MEM-HLD-04": [
        "Did Ben update the Net Retention Rate after investigating the churn numbers?",
        "What is our confirmed NRR metric for the board deck?",
        "What did the churn dashboard NRR end up being after the calculation bug was resolved?",
        "What's our net retention rate percentage right now?"
    ],
    "MEM-HLD-05": [
        "Did we ship the driver stop reordering bug fix to Pinecrest yet?",
        "Has Dave Morales been notified that the stop reordering bug is resolved?",
        "What's the status of the stop reordering fix that Pinecrest requested?",
        "Is the driver app stop reordering fix ready for Pinecrest to test?"
    ],
    "MEM-HLD-06": [
        "Why did Dana change the route line color palette, and which color had an issue?",
        "What was Sarah Kim's feedback on the new eight-color route lines Dana proposed?",
        "Why was the coral route color removed from the v2 route planner design?",
        "What changes did Dana make to the map colors to address colorblindness and traffic overlay?"
    ],
    "MEM-HLD-07": [
        "Where does Copperline Freight stand in our Q3 sales forecast?",
        "How many trucks does Copperline Freight have, and are they in our commit numbers?",
        "Did John include Copperline Freight in the board forecast during pipeline review?",
        "What was reported about the Copperline Freight inbound lead during Tuesday's pipeline meeting?"
    ],
    "MEM-HLD-08": [
        "What policy did Rachel announce regarding contract redlines in Slack?",
        "Do we need legal review from Rachel before sending out agreement redlines?",
        "What did Rachel Gomez say in general about customer contracts?",
        "Are sales reps allowed to send customer redlines without Rachel's approval?"
    ],
    "MEM-HLD-09": [
        "What are the details for the upcoming Q3 board meeting?",
        "Where is the board meeting taking place next Wednesday, and at what time?",
        "When and where are we meeting the board next week?",
        "What time does our board meeting at Foundry Ridge start?"
    ],
    "MEM-HLD-10": [
        "Can we hire an outside designer for the marketing website?",
        "What did John tell Alex about bringing on a contractor to redesign the marketing site?",
        "Did John give the green light for a marketing site design contractor?",
        "Are we allowed to use a contractor for marketing site design work?"
    ],
    "MEM-HLD-11": [
        "What is the guest Wi-Fi network password for our office?",
        "Can you tell me the Brightline office wifi password?",
        "What's the credentials for connecting to the office internet?",
        "Where can I find our corporate wifi password?"
    ],
    "MEM-HLD-12": [
        "Where does Alex Rivera live?",
        "What is Alex's home address?",
        "Can you look up Alex Rivera's residential address?",
        "What street does Alex live on?"
    ]
}

ADDITIONAL_HOLDOUT_QUESTIONS = [
    {
        "id": "MEM-HLD-01",
        "question": "When is the renewal check-in meeting with Pinecrest Carriers scheduled?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "calendar_travel",
        "answerable": True,
        "gold_answer": "Tuesday, September 22, 2026 from 10:00 to 10:30 AM (with Dave Morales and Marcus Webb).",
        "rubric": ["Tuesday Sep 22", "10:00 to 10:30am"],
        "key_terms": [["sep 22", "september 22", "10:00", "tuesday"]],
        "evidence": ["CAL-F-21", "EM-F-040"],
        "also_supports": ["EM-F-008", "MTG-0915-SALESPIPE#0008"],
        "storyline": "pinecrest_renewal",
        "needed": [["CAL-F-21", "EM-F-040"]]
    },
    {
        "id": "MEM-HLD-02",
        "question": "What did we decide about Jordan Ellis after the recruiting debrief?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "ownership",
        "answerable": True,
        "gold_answer": "Move Jordan to the final round for a second system design session with Sarah Kim and Alex, plus a culture/values conversation with John Okafor, scheduled for Monday Sep 21 or Tuesday Sep 22 before the board meeting.",
        "rubric": ["Move to final round", "Second system design session / culture chat with John"],
        "key_terms": [["final round", "final", "second system design"]],
        "evidence": ["MTG-0917-RECRUIT"],
        "also_supports": ["MTG-0917-RECRUIT#0028", "MTG-0917-RECRUIT#0035", "MTG-0917-RECRUIT#0037", "MTG-0917-RECRUIT#0038"],
        "storyline": "recruiting_debrief",
        "needed": [["MTG-0917-RECRUIT#0028", "MTG-0917-RECRUIT#0035", "MTG-0917-RECRUIT#0037", "MTG-0917-RECRUIT#0038"]]
    },
    {
        "id": "MEM-HLD-03",
        "question": "What were the interview ratings for Jordan Ellis from Ben and Sarah?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "disagreement",
        "answerable": True,
        "gold_answer": "Ben gave a strong yes (impressed by Postgres vacuum and bloat knowledge), while Sarah Kim gave a lean no on system design (live ETA updates).",
        "rubric": ["Ben was strong yes", "Sarah was lean no on system design"],
        "key_terms": [["strong yes", "yes"], ["lean no", "no"]],
        "evidence": ["MTG-0917-RECRUIT"],
        "also_supports": ["MTG-0917-RECRUIT#0009", "MTG-0917-RECRUIT#0011", "MTG-0917-RECRUIT#0015", "MTG-0917-RECRUIT#0019"],
        "storyline": "recruiting_feedback",
        "needed": [["MTG-0917-RECRUIT#0009", "MTG-0917-RECRUIT#0011"], ["MTG-0917-RECRUIT#0015", "MTG-0917-RECRUIT#0019"]]
    },
    {
        "id": "MEM-HLD-04",
        "question": "What is our current Net Retention Rate (NRR)?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "correction",
        "answerable": True,
        "gold_answer": "112%. The churn dashboard initially showed 118%, but Ben Carter confirmed on Sep 16 that it had double-counted an expansion and fixed it.",
        "rubric": ["NRR is 112%", "Does not report 118% as current"],
        "key_terms": [["112", "112%"]],
        "stale": [["118%"]],
        "evidence": ["SL-DM-BA-0916-1", "DCT-0917-06"],
        "also_supports": ["DCT-0908-01"],
        "storyline": "nrr_metric",
        "needed": [["SL-DM-BA-0916-1", "DCT-0917-06"]]
    },
    {
        "id": "MEM-HLD-05",
        "question": "Where does the stop reordering fix for Pinecrest Carriers stand?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "commitment_status",
        "answerable": True,
        "gold_answer": "The fix is in staging as of Friday Sep 18; Alex emailed Dave Morales asking Pinecrest's team to test it on Monday's routes (Priya also completed Linear issue RP-198 on Sep 8).",
        "rubric": ["Fix is in staging", "Asked Dave / Pinecrest to test on Monday"],
        "key_terms": [["staging", "test", "monday"]],
        "evidence": ["DCT-F-035", "DCT-F-004"],
        "also_supports": ["SL-F-0007"],
        "storyline": "stop_reorder_bug",
        "needed": [["DCT-F-035", "DCT-F-004"]]
    },
    {
        "id": "MEM-HLD-06",
        "question": "Why did Dana reduce the route line colors to 8, and what color did Sarah Kim ask her to replace?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "single_fact",
        "answerable": True,
        "gold_answer": "Dana reduced the palette from 10 to 8 colors for colorblind accessibility and grayscale readability. Sarah Kim asked her to replace coral because it collided with the traffic congestion layer, which Dana agreed to swap with purple.",
        "rubric": ["Colorblind accessibility / grayscale", "Coral collided with traffic layer"],
        "key_terms": [["colorblind", "accessibility"], ["coral", "traffic"]],
        "evidence": ["MTG-0911-DESIGN"],
        "also_supports": ["MTG-0911-DESIGN#0060", "MTG-0911-DESIGN#0062", "MTG-0911-DESIGN#0066", "MTG-0911-DESIGN#0068", "MTG-0911-DESIGN#0069"],
        "storyline": "design_palette",
        "needed": [["MTG-0911-DESIGN#0060", "MTG-0911-DESIGN#0062"], ["MTG-0911-DESIGN#0066", "MTG-0911-DESIGN#0068", "MTG-0911-DESIGN#0069"]]
    },
    {
        "id": "MEM-HLD-07",
        "question": "What is Copperline Freight, and what is its status in the board forecast?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "single_fact",
        "answerable": True,
        "gold_answer": "Copperline Freight is an inbound lead with 60 trucks (mostly dry van out of Sacramento). In the Sep 15 pipeline review, John Okafor kept it as pipeline, not in the board commit forecast.",
        "rubric": ["60 trucks / Sacramento", "Kept in pipeline, excluded from commit forecast"],
        "key_terms": [["60 trucks", "sacramento", "pipeline"]],
        "evidence": ["MTG-0915-SALESPIPE#0042", "MTG-0915-SALESPIPE#0051", "MTG-0915-SALESPIPE#0086"],
        "also_supports": ["MTG-0915-SALESPIPE#0092"],
        "storyline": "sales_pipeline",
        "needed": [["MTG-0915-SALESPIPE#0042", "MTG-0915-SALESPIPE#0051"]]
    },
    {
        "id": "MEM-HLD-08",
        "question": "What instructions did Rachel Gomez give about customer contract redlines?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "single_fact",
        "answerable": True,
        "gold_answer": "Route any customer contract redlines through her before sending, even small ones (posted to Slack general on Sep 14).",
        "rubric": ["Route all contract redlines through her before sending", "Even small ones"],
        "key_terms": [["route", "redlines", "rachel"]],
        "evidence": ["SL-F-0120"],
        "also_supports": [],
        "storyline": "legal_contracting",
        "needed": [["SL-F-0120"]]
    },
    {
        "id": "MEM-HLD-09",
        "question": "Where and when is the Q3 board meeting taking place?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "temporal",
        "answerable": True,
        "gold_answer": "Wednesday, September 23, 2026 from 9:00 AM to 12:00 PM at Foundry Ridge (1 Market St), hosted by Tom.",
        "rubric": ["Wednesday Sep 23, 9am–12pm", "Foundry Ridge"],
        "key_terms": [["sep 23", "september 23", "foundry ridge", "9"]],
        "evidence": ["CAL-BOARD", "MTG-0910-1ON1#0035"],
        "also_supports": ["CAL-F-25"],
        "storyline": "board_preparation",
        "needed": [["CAL-BOARD", "MTG-0910-1ON1#0035"]]
    },
    {
        "id": "MEM-HLD-10",
        "question": "Did John approve hiring a marketing contractor?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "conditional",
        "answerable": True,
        "gold_answer": "Conditionally open to it: John told Alex in their Sep 10 1:1 to find someone and send a proposal for a small scope (a few weeks of work, not a retainer).",
        "rubric": ["Told Alex to send a proposal", "Small scope / a few weeks, not retainer"],
        "key_terms": [["proposal", "small scope", "marketing"]],
        "evidence": ["MTG-0910-1ON1#0061", "MTG-0910-1ON1#0065"],
        "also_supports": ["MTG-0910-1ON1#0049"],
        "storyline": "marketing_contractor",
        "needed": [["MTG-0910-1ON1#0061", "MTG-0910-1ON1#0065"]]
    },
    {
        "id": "MEM-HLD-11",
        "question": "What is the Wi-Fi password for the Brightline office?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "abstention",
        "answerable": False,
        "gold_answer": "Not in memory: nothing in the data covers this.",
        "rubric": ["Says it has no record / doesn't know", "Does not invent an answer"],
        "key_terms": [],
        "evidence": [],
        "also_supports": [],
        "storyline": "not_in_memory",
        "needed": []
    },
    {
        "id": "MEM-HLD-12",
        "question": "What is Alex Rivera's home address?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "category": "abstention",
        "answerable": False,
        "gold_answer": "Not in memory: nothing in the data covers this.",
        "rubric": ["Says it has no record / doesn't know", "Does not invent an answer"],
        "key_terms": [],
        "evidence": [],
        "also_supports": [],
        "storyline": "not_in_memory",
        "needed": []
    }
]


def build():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVALS_DIR / "memory_train.jsonl", "r", encoding="utf-8") as f:
        original_records = [json.loads(line) for line in f if line.strip()]

    # Combine original training records with additional independent holdout questions
    original_records = list(original_records) + list(ADDITIONAL_HOLDOUT_QUESTIONS)

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
