# Document Q&A Agent — Clarifying Questions

**Date:** 2026-04-07
**Purpose:** Gather requirements before redesigning the document-qa-agent to handle general questions, complex queries, and behave like a full-context AI platform (Claude web-like experience).

**Instructions:** Please answer each question. Write your answer below each question or reply with the question numbers.

---

## 1. Document Characteristics

### Q1.1 — Document Size
What is the typical page count of documents uploaded to this agent?
- (A) Small: 1–20 pages (scope docs, product data sheets)
- (B) Medium: 20–100 pages (specifications, contracts)
- (C) Large: 100–500+ pages (full project manuals, combined specs)
- (D) Mix of all the above

**Answer:**  Mix of all above, so it should be scalable

### Q1.2 — Document Volume Per Session
How many documents does a typical user upload in a single session?
- (A) 1 document at a time
- (B) 2–5 related documents
- (C) 5–20 documents (full project package)
- (D) Varies widely

**Answer:** Varies widely

### Q1.3 — Document Formats Priority
Rank the most common formats your users upload (1 = most common):
- [ ] PDF (text-based specs, submittals)
- [ ] PDF (scanned/image-based drawings, floor plans)
- [ ] Word (.docx) — scope documents, reports
- [ ] Excel (.xlsx) — schedules, data sheets
- [ ] Images (.jpg/.png) — site photos, markups
- [ ] Other: ___________

**Answer:** All of the above

### Q1.4 — Construction Drawing Handling
For PDF floor plans and construction drawings (like A101-FLOOR-PLAN.pdf), the current agent uses Vision API to extract content. Is this working well enough, or do you need improvements here too?
- (A) Vision extraction is working fine — focus on text document Q&A
- (B) Vision extraction needs improvement too — drawings are a priority
- (C) Drawings are secondary — fix text document Q&A first, drawings later

**Answer:** - (C) Drawings are secondary — fix text document Q&A first, drawings later


---

## 2. Query Behavior & Expectations

### Q2.1 — General vs Specific Questions
From the screenshot, the agent fails on general questions like "what are the Business Objectives?" but works on specific ones. What types of general questions should work?
- (A) Summarization: "Summarize this document", "What is this document about?"
- (B) Broad topic extraction: "What are the business objectives?", "List all requirements"
- (C) Cross-section synthesis: "How does fire alarm relate to access control?"
- (D) All of the above — it should answer ANY question about the document

**Answer:** - (D) All of the above — it should answer ANY question about the document


### Q2.2 — Multi-Document Questions
Should the agent be able to answer questions that span across multiple uploaded documents?
- (A) Yes — "Compare the electrical scope with the mechanical scope"
- (B) Only within the same document
- (C) Yes, but clearly cite which document each fact comes from

**Answer:** - (C) Yes, but clearly cite which document each fact comes from


### Q2.3 — Conversation Depth
How many follow-up questions does a typical user ask in a session?
- (A) 1–3 quick questions
- (B) 5–10 deep-dive questions
- (C) 10–30+ extended research session
- (D) Varies — should support all depths

**Answer:** - (D) Varies — should support all depths


### Q2.4 — Response Style
How should the agent respond?
- (A) Concise bullet points (current style)
- (B) Detailed paragraphs with full context
- (C) Adaptive — short for simple questions, detailed for complex ones
- (D) Match the user's question style

**Answer:** - (D) Match the user's question style


---

## 3. Technical & Cost Constraints

### Q3.1 — LLM Model
Currently using `gpt-4o-mini`. For better general question handling, we may need a more capable model. What's acceptable?
- (A) Stay with gpt-4o-mini (cheapest, but weakest reasoning)
- (B) Upgrade to gpt-4o (better reasoning, ~10x cost increase)
- (C) Switch to Claude (Sonnet 4 or Opus 4) for better document understanding
- (D) Hybrid: use gpt-4o-mini for simple queries, escalate to gpt-4o/Claude for complex ones
- (E) No preference — recommend the best option

**Answer:** - (E) No preference — recommend the best option but use only OpenAI llms.


### Q3.2 — Cost Tolerance
Full-document context (Claude web-like) means sending the entire document to the LLM every query, which costs more tokens. What's your monthly API budget tolerance per user?
- (A) Keep costs minimal (~$0.001–0.01 per query) — current level
- (B) Moderate ($0.01–0.10 per query) — acceptable for better quality
- (C) No strict limit — quality is the priority
- (D) Need a cost estimate before deciding

**Answer:** - (C) No strict limit — quality is the priority


### Q3.3 — Response Latency
What's acceptable response time?
- (A) Under 3 seconds (current streaming starts in ~200ms)
- (B) Under 10 seconds is fine for better quality
- (C) Under 30 seconds acceptable for complex multi-page analysis
- (D) Streaming is fine — as long as it starts quickly, total time doesn't matter

**Answer:** - (D) Streaming is fine — as long as it starts quickly, total time doesn't matter


### Q3.4 — Concurrent Users
How many users will use this agent simultaneously?
- (A) 1–5 (internal team)
- (B) 5–20 (department level)
- (C) 20–100 (company-wide)
- (D) 100+ (customer-facing)

**Answer:** 100+ (customer-facing)

---

## 4. Frontend & Integration

### Q4.1 — Frontend UI
The screenshot shows a UI labeled "Drawings Agent" with "Contract Document" label and a chat interface. Is this:
- (A) The Streamlit app included in the repo
- (B) A separate React/Vue frontend (IFS platform)
- (C) A third-party chat widget
- (D) Other: ___________

**Answer:** - (B) A separate React/Vue frontend (IFS platform)


### Q4.2 — API Contract
If the frontend is separate, can we change the API response format, or must we maintain backward compatibility with the current `/api/converse` and `/api/chat` endpoints?
- (A) Can change freely — we control the frontend
- (B) Must keep existing endpoints, can add new ones
- (C) Strict backward compatibility required

**Answer:** - (A) Can change freely — we control the frontend


### Q4.3 — Session Continuity
Should a user be able to return to a previous session (days later) and continue asking questions about documents they uploaded before?
- (A) Yes — sessions should persist for days/weeks
- (B) 24-hour sessions are fine (current behavior)
- (C) Sessions should last as long as the project is active

**Answer:** - (A) Yes — sessions should persist for days/weeks


---

## 5. Quality & Safety

### Q5.1 — Hallucination Guard
The current hallucination guard (token-overlap scoring) triggers false positives on general answers because general summaries naturally use different words than the source. Should we:
- (A) Replace with a smarter guard (LLM-based faithfulness check)
- (B) Relax the threshold significantly for general questions
- (C) Remove the guard entirely — trust the LLM
- (D) Keep the guard but only for specific factual claims, not summaries

**Answer:** - (A) Replace with a smarter guard (LLM-based faithfulness check) - (B) Relax the threshold significantly for general questions



### Q5.2 — Out-of-Context Handling
When a user asks something NOT in the document, should the agent:
- (A) Strictly refuse: "This is not in the document" (current behavior)
- (B) Attempt to answer from document context + general knowledge, clearly labeled
- (C) Refuse but suggest what topics ARE in the document
- (D) Answer from general knowledge with a disclaimer

**Answer:** - (A) Strictly refuse: "This is not in the document" (current behavior)


### Q5.3 — Citation Requirements
How important is source citation?
- (A) Critical — every claim must cite page/section
- (B) Important — cite when possible, but summaries don't need per-sentence citations
- (C) Nice to have — not a hard requirement

**Answer:** - (A) Critical — every claim must cite page/section


---

## 6. Deployment & Operations

### Q6.1 — Sandbox Testing
You mentioned sandbox VM (54.197.189.113). Should the redesigned agent be:
- (A) Tested on sandbox first, then moved to production (13.217.22.125)
- (B) Only deployed to sandbox for now
- (C) Deployed to both simultaneously

**Answer:** - (A) Tested on sandbox first, then moved to production (13.217.22.125)


### Q6.2 — Rollback Strategy
If the new version has issues, should we:
- (A) Keep the old agent running in parallel (different port/route)
- (B) Feature flag to switch between old/new behavior
- (C) Replace entirely — we'll test thoroughly first

**Answer:** - (B) Feature flag to switch between old/new behavior


### Q6.3 — Monitoring
What monitoring do you need?
- (A) Basic logs (current level)
- (B) Per-query metrics dashboard (latency, cost, groundedness)
- (C) Alerting on failures/high costs
- (D) All of the above

**Answer:** - (D) All of the above


---

## 7. Demo Questions Validation

You provided these demo questions for the electrical scope document. Please confirm these should ALL work after the redesign:

1. "What happens to access-controlled doors during a fire alarm event?"
2. "What components are required for fire alarm integration (relays, interface modules)?"
3. "What testing is required after integration with fire alarm systems?"

**Answer:** yes

**Additional test scenarios that should also work:**
4. "Summarize this document"
5. "What is the scope of work?"
6. "List all the systems mentioned in this document"
7. "What are the key requirements?"
8. "Compare the electrical requirements with safety requirements"

**Answer:** yes

**Should all 8 of these work? Any others you want to add?**

**Answer:** Yes

---

## Next Steps

Once you answer these questions, I will:
1. Propose 2–3 architectural approaches with trade-offs
2. Present a detailed design for your approval
3. Create a phased implementation plan
4. Execute development with rigorous testing
5. Deploy to sandbox VM → production

**Please answer and return this document.**
