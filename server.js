require("dotenv").config();
const express = require("express");
const cors = require("cors");
const axios = require("axios");
const multer = require("multer");
const mammoth = require("mammoth");

const app = express();
app.use(cors({ origin: "*" }));
app.use(express.json({ limit: "10mb" }));
app.use((req, res, next) => { req.setTimeout(300000); res.setTimeout(300000); next(); });

const ANTHROPIC = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
const MODEL = "claude-sonnet-4-20250514";
const GPTZERO_KEY = process.env.GPTZERO_KEY;
const ANTHROPIC_KEY = process.env.ANTHROPIC_KEY;

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

function headers() {
  return {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "anthropic-beta": "prompt-caching-2024-07-31"
  };
}

async function claude(systemText, userMessage, useWebSearch) {
  const body = {
    model: MODEL,
    max_tokens: 4000,
    system: [{ type: "text", text: systemText, cache_control: { type: "ephemeral" } }],
    messages: [{ role: "user", content: userMessage }]
  };
  if (useWebSearch) body.tools = [{ type: "web_search_20250305", name: "web_search" }];
  const res = await axios.post(ANTHROPIC, body, { headers: headers() });
  return (res.data.content || []).filter(b => b.type === "text").map(b => b.text).join("\n").trim();
}

const TEXT_TYPES = {
  article: { name: "Article", format: `FORMAT: 1. Headline 2. Introduction — hook, introduce topic 3. Body paragraphs — one point each, facts and examples 4. Conclusion — summarize, leave reader thinking. TONE: formal or semi-formal, clear, organized.`, searchGuidance: "factual reporting, expert opinions, and statistics that inform a general audience" },
  blog: { name: "Blog Post", format: `FORMAT: 1. Catchy title 2. Opening hook — personal, engaging 3. Introduction 4. Main body — ideas, tips, anecdotes 5. Conclusion — final thought or question. TONE: informal, friendly, personal. Use "I" and "you".`, searchGuidance: "personal stories, practical tips, and conversational takes" },
  oped: { name: "Op-Ed", format: `FORMAT: 1. Headline 2. Introduction — hook, state issue, present opinion 3. Argument 1 with evidence 4. Argument 2 with evidence 5. Counterargument addressed 6. Conclusion — reinforce opinion, strong ending. TONE: confident, persuasive, assertive.`, searchGuidance: "evidence, statistics, and expert opinions that support a persuasive argument, plus one counterargument" },
  speech: { name: "Speech", format: `FORMAT: 1. Greeting and opening — grab attention 2. Introduction — topic and purpose 3. Main points — 2 to 3 key ideas with examples 4. Conclusion — memorable final line. TONE: written to be heard. Short sentences, repetition, emotional appeal, direct audience address.`, searchGuidance: "powerful quotes, emotional stories, and compelling facts that resonate when spoken aloud" },
  essay: { name: "Essay", format: `FORMAT: 1. Introduction — topic, clear thesis 2. Body paragraph 1 — point, evidence, explanation 3. Body paragraph 2 4. Body paragraph 3 5. Conclusion — summarize, restate thesis freshly. TONE: formal and academic. Analytical, precise vocabulary.`, searchGuidance: "academic evidence and logical arguments for a structured analytical essay" },
  review: { name: "Review", format: `FORMAT: 1. Title 2. Introduction — what is reviewed, brief context 3. Description 4. Evaluation — strengths and weaknesses 5. Conclusion — final judgment and recommendation. TONE: semi-formal, lively, balanced.`, searchGuidance: "detailed descriptions, critic opinions, and evaluative takes" },
  report: { name: "Report", format: `FORMAT: 1. Title 2. Introduction — purpose 3. Background and context 4. Findings 5. Analysis 6. Recommendations if needed 7. Conclusion. TONE: formal, objective, neutral. Use plain text subheadings.`, searchGuidance: "data, research findings, official statistics, and expert analysis" }
};

app.get("/health", (req, res) => res.json({ ok: true }));

// NEW: Analyze prompt before generation
app.post("/analyze", async (req, res) => {
  const { prompt, textType } = req.body;
  if (!prompt) return res.status(400).json({ error: "Missing prompt" });

  const system = `You are an AI writing quality analyst. Your job is to estimate how well an AI writing system can complete a given prompt by finding real human-written sources online.

The system works by:
1. Searching for human-written content in foreign languages on the topic
2. Translating those passages literally into English
3. Stitching them together with minimal AI connective tissue

CRITICAL RULE: Judge based on the UNDERLYING TOPIC, not the document format or framing.
- An assignment brief asking students to research climate change = research_based (the topic is climate change)
- A school task asking for a report on migration = research_based (the topic is migration)
- A template or brief that asks for research on any well-known topic = research_based
- Only flag as personal/impossible if the actual content required is personal memories, private information, or analysis of an unprovided document

The system performs BEST when the underlying topic is:
- Widely covered by human journalists, academics, bloggers, researchers
- A known global issue, historical event, scientific subject, cultural topic, or current event
- Something that exists in multiple languages online

The system performs POORLY when the actual content required is:
- Personal experiences, memories, or private information only the user has
- Analysis of a specific private document, poem, or text not provided
- Purely fictional/creative with no factual basis to research

Categories:
- "research_based": Underlying topic is well-covered online in multiple languages by journalists, academics, bloggers. System will perform at 85%+. Examples: climate change, migration, poverty, conflict, history, science, culture, global issues, current events, any assignment asking for research on known real-world topics.
- "semi_personal": Topic has personal elements but general themes are researchable. System will get 65-84%. Examples: personal essay on universal themes like resilience or family, opinion pieces where the argument can be sourced even if the voice is personal.
- "highly_personal": Content requires specific personal information OR is a close literary/textual analysis of a specific known work. System will get 30-64%. Examples: "write about how my grandmother's death changed me", literary analysis of a specific poem or novel (e.g. Ozymandias, Hamlet, To Kill a Mockingbird), analysis of a specific piece of art or music, any task where the content must come from deep reading of one specific text rather than broad research.
- "impossible": Almost entirely personal/private with no researchable component. System will get under 30%. Examples: analysis of a private unprovided document, purely fictional story with no factual basis, writing that requires information only the user holds.

IMPORTANT: Literary analysis assignments (poetry analysis, novel analysis, close reading tasks) should ALWAYS be classified as "highly_personal" even if the text being analyzed is famous and well-known. The reason: our system sources from foreign language articles and translates them. This works for factual research topics but NOT for literary analysis, where the content must come from close reading of specific lines and structural features of one text. Foreign language articles about Ozymandias will not produce good literary analysis.

Return ONLY a JSON object:
{
  "estimatedHumanPct": <number between 5 and 98>,
  "category": "research_based" | "semi_personal" | "highly_personal" | "impossible",
  "reasoning": "<one sentence explaining why>"
}

No preamble, no explanation.`;

  try {
    const raw = await claude(system, "Analyze this prompt for a " + (textType || "essay") + ":\n\n" + prompt);
    const parsed = JSON.parse(raw.replace(/```json|```/g, "").trim());
    res.json(parsed);
  } catch (err) {
    // If analysis fails, default to proceeding normally
    res.json({ estimatedHumanPct: 85, category: "research_based", reasoning: "Could not analyze — proceeding normally." });
  }
});

app.post("/extract", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });
  const mime = req.file.mimetype;
  const buffer = req.file.buffer;
  try {
    let text = "";
    if (mime === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
      const result = await mammoth.extractRawText({ buffer });
      text = result.value;
    } else {
      text = buffer.toString("utf-8");
    }
    res.json({ text: text.trim() });
  } catch (err) {
    res.status(500).json({ error: "Could not extract text: " + err.message });
  }
});

app.post("/generate", async (req, res) => {
  const { prompt, citations, textType } = req.body;
  if (!prompt) return res.status(400).json({ error: "Missing prompt" });

  const typeConfig = TEXT_TYPES[textType] || TEXT_TYPES.oped;
  const citationRule = citations
    ? "Add a short in-text citation in parentheses after each passage e.g. (Source Name, Year). Do NOT mention the language it was translated from. Always include a full SOURCES section at the end."
    : "Do NOT include any in-text citations in the body. Always include a full SOURCES section at the end listing every URL used.";

  const isDocumentBrief = prompt.length > 500 &&
    /rubric|criterion|assignment|task|report structure|research question|marks|assessment|bibliography|learning objective|ATL|MYP|IB/i.test(prompt);

  let system;

  if (isDocumentBrief) {
    system = `You are a Research Curator who writes structured pieces by sourcing real human-written content in foreign languages and translating it literally into English.

OUTPUT LANGUAGE — NON-NEGOTIABLE: Every word of the final piece must be in English. No exceptions. You are translating foreign sources INTO English, not reproducing them. If you retrieved a Spanish article, a French blog post, a German study — translate every sentence into English before you write it down. Do not output a single word in any language other than English. If you catch yourself writing in Spanish, French, German, or any other language, stop and translate it immediately.

A document brief or assignment rubric has been provided. Your job:
1. READ the brief fully and understand ALL requirements — structure, word count, sections, research question format, citation style, criteria.
2. IDENTIFY the topic and specific location/community to focus on. If the user specified one in their additional instructions, use that. If not, pick the most interesting and well-documented option from those listed.
3. FORMULATE a proper research question following the format in the brief exactly (e.g. "How and to what extent does X affect Y, and what responses are most effective?").
4. FOLLOW the required structure exactly — use every section heading in the brief, in the right order, with appropriate content in each.
5. MEET all listed requirements — minimum sources, statistics, word count, section content, everything.
6. SOURCE everything from foreign language human-written content, translated literally into English.

SOURCING:
- Choose 3-4 languages with rich content on the specific topic and chosen location
- Find 5-7 sources covering background, local impact, responses, and evaluation
- Extract 3-5 consecutive sentences from each source
- Translate literally — preserve source rhythm, do not over-smooth into perfect English
- MANDATORY: every sentence must be in English. Read through before outputting and translate anything remaining.

CONNECTIVE TISSUE between source blocks — keep minimal:
- Sharp, slightly impatient expert voice
- NO m-dashes, NO colons in transitions
- BANNED: delve, elevate, underscore, testament, navigate, foster, tapestry, unlock, robust, inherently
- USE: Then there is the fact that / What is wild is / But then you look at

CRITICAL OUTPUT RULES:
- NEVER include the original foreign language text. Translate it and include ONLY the English version.
- NEVER mention that you searched foreign language sources, gathered background material, or describe your process in any way. The reader should never know how the piece was created.
- NEVER write phrases like "A French analysis reveals:", "Spanish sources indicate:", "According to a German report:", "I started gathering", "This analysis requires". Just state the content as fact.
- If a citation is needed, format it as (Source Name, Year) only — never mention the language.
- The piece must read as if written by a single knowledgeable human author. No trace of the sourcing process should appear anywhere.

${citationRule}
Do NOT bold any text. No markdown # headers. No footnote numbers.
Output ONLY the finished piece with all required sections and SOURCES. Nothing else.`;
  } else {
    system = `You are a Research Curator writing a ${typeConfig.name}. Find human-written source material in foreign languages, translate it literally into English, and assemble the piece from those translations.

OUTPUT LANGUAGE — NON-NEGOTIABLE: Every word of the final piece must be in English. No exceptions. You are translating foreign sources INTO English, not reproducing them. If you retrieved a Spanish article, a French blog post, a German study — translate every sentence into English before you write it down. Do not output a single word in any language other than English. The reader will never see the original sources. They see only your English translation. If you catch yourself writing in Spanish, French, German, or any other language, stop and translate it immediately.

LANGUAGE STRATEGY:
1. Think about which languages have the richest human-written content on this specific topic. Choose 3-4 languages that genuinely make sense.
2. Search in those languages for human-written articles, academic pieces, or journalism.
3. Find 5-7 sources total.
4. Extract the longest useful consecutive block you can from each source — up to 5-6 sentences from one source if highly relevant.
5. Translate each block into English:
   - Stay close to the original sentence structure to preserve the foreign rhythm.
   - When the source is scientific or academic, translate technical jargon into plain language appropriate for the text type.
   - When a sentence uses a passive construction that hides who is doing what, rewrite it to make the agent clear.
   - When a sentence references a mechanism or process without explaining it, either explain it in plain terms or cut it.
   - When a statistic needs context to make sense, add minimal necessary context.
   - Never leave a sentence that would confuse a reader with no background in the topic.
   - Do NOT over-smooth into polished English — preserve the rhythm of the source language.
6. Translate EVERY extracted block into English before assembling. Do not leave any foreign language text in the final output — not a single word. Every sentence in the finished piece must be in English. If you find yourself including a foreign language sentence, stop and translate it first.
7. Assemble the piece from translated blocks.
8. MANDATORY SELF-CHECK before outputting: Read through your entire draft. If you find ANY sentence that is not in English — any word in French, Spanish, German, Portuguese, Chinese, Japanese, or any other language — translate it immediately. Do not output the piece until every single sentence is in English. This step is required every time.

CONNECTIVE TISSUE — keep it minimal (max 1-2 short sentences between blocks):
- Sharp, slightly impatient human expert voice.
- NO m-dashes. NO symmetry patterns. NO colons in connective sentences.
- BANNED: delve, elevate, underscore, testament, navigate, foster, tapestry, unlock, robust, inherently, comprehensive, imperative, multifaceted
- NO: Furthermore, In addition, Subsequently
- USE: Then there is the fact that / What is wild is / But then you look at / This suggests that

TEXT TYPE: ${typeConfig.name}
${typeConfig.format}

WORD COUNT — CRITICAL: Hit the word count in the prompt accurately. Do not exceed by more than 20 words. Do not fall short by more than 20 words.

BEFORE OUTPUTTING, fix:
- Any named individual the reader has not met — rewrite without the name
- Any phrase referencing a website's own tools or database — rewrite without it
- Any numbered footnote markers — delete them
- Any sentence that only makes sense in the original source context
- Any incomplete ending — write a proper conclusion for the text type
- Any two consecutive statistics that appear to contradict each other — explain or cut one



CRITICAL OUTPUT RULES:
- NEVER include the original foreign language text. Translate it and include ONLY the English version.
- NEVER mention that you searched foreign language sources, gathered background material, or describe your process in any way. The reader should never know how the piece was created.
- NEVER write phrases like "A French analysis reveals:", "Spanish sources indicate:", "According to a German report:", "I started gathering", "This analysis requires". Just state the content as fact.
- If a citation is needed, format it as (Source Name, Year) only — never mention the language.
- The piece must read as if written by a single knowledgeable human author. No trace of the sourcing process should appear anywhere.

${citationRule}
Do NOT bold any text. No markdown # headers. No footnote numbers.
Output ONLY the finished piece and SOURCES section. Nothing else.`;
  }

  try {
    let draft = await claude(system, prompt, true);
    draft = draft.replace(/^#{1,6}\s+/gm, "").replace(/\*\*/g, "").replace(/\[\d+\]/g, "").trim();

    // Post-process: enforce English paragraph by paragraph in parallel
    // Claude detects the language of each paragraph itself — no regex, no missed cases
    const paraLangSystem = `If the text below is already in English, return it word-for-word, unchanged. If it is in any other language, translate it into English. Output only the result — no explanation, no commentary, nothing else.`;

    const paragraphs = draft.split(/\n\n+/);
    const fixedParagraphs = await Promise.all(paragraphs.map(async (para) => {
      if (para.trim().length < 15) return para; // skip blank lines / very short separators
      try {
        const result = await claude(paraLangSystem, para);
        return result.replace(/^#{1,6}\s+/gm, "").replace(/\*\*/g, "").trim();
      } catch (err) {
        console.error("Para lang check failed:", err.message);
        return para;
      }
    }));
    draft = fixedParagraphs.join("\n\n");

    // Third pass: fix translation artifacts — broken syntax, foreign word order, meaningless fragments
    const artifactCleanupSystem = `You are a copy editor fixing translation artifacts in a text assembled from foreign language sources. Find and fix only sentences that are broken or unnatural due to bad translation. Do not touch sentences that read naturally.

Fix ONLY these problems:
1. Foreign word order that makes no sense in English — rewrite that sentence in natural English with the same meaning
2. Completely meaningless or garbled fragments — delete them
3. Literal translations of idioms that produce nonsense in English — replace with the natural English equivalent
4. Wrong word choices from translation errors (e.g. "internal courts" → "domestic courts", "does not provide any paragraph" → "contains no provision") — fix the word only
5. Bureaucratic copy-paste from UN resolutions or legal documents that reads like no human wrote it — simplify to plain English

Do NOT rewrite sentences that already read naturally. Do NOT change facts, statistics, or claims. Do NOT add content.

Output the complete corrected text. No commentary.`;

    try {
      const artifactCleaned = await claude(artifactCleanupSystem, draft);
      draft = artifactCleaned.replace(/^#{1,6}\s+/gm, "").replace(/\*\*/g, "").trim();
    } catch (err) {
      console.error("Artifact cleanup failed:", err.message);
    }

    res.json({ draft });
  } catch (err) {
    res.status(500).json({ error: "Generate error: " + (err.response?.data?.error?.message || err.message) });
  }
});


app.post("/scan", async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "Missing text" });
  try {
    const response = await axios.post(
      "https://api.gptzero.me/v2/predict/text",
      { document: text },
      { headers: { "Content-Type": "application/json", "x-api-key": GPTZERO_KEY } }
    );
    const doc = response.data.documents?.[0];
    if (!doc) return res.status(500).json({ error: "No document in GPTZero response" });
    res.json({
      overallAiProb: doc.completely_generated_prob ?? 0,
      sentences: (doc.sentences || []).map(s => ({ text: s.sentence, aiProb: s.generated_prob ?? 0 }))
    });
  } catch (err) {
    res.status(500).json({ error: "GPTZero error: " + (err.response?.data?.message || err.message) });
  }
});

app.post("/humanize", async (req, res) => {
  const { draft, flaggedSentences, textType, citations } = req.body;
  if (!draft || !flaggedSentences?.length) return res.status(400).json({ error: "Missing data" });

  const typeConfig = TEXT_TYPES[textType] || TEXT_TYPES.oped;
  const citationNote = citations ? "Preserve all existing in-text citations." : "Do not add any in-text citations.";

  const system = `You are a humanization editor. You receive a full draft and a list of AI-flagged sentences. For each flagged sentence, follow this exact process:

STEP 1 — SEARCH FOR A REPLACEMENT FIRST:
Search the web in foreign languages (choose the language most likely to have good human-written content on this specific topic) for a human-written source that covers the same claim or fact as the flagged sentence. Try to find 1-3 consecutive sentences from a real human author that express the same idea.

If you find a good source:
- Extract the relevant sentences verbatim from the foreign language source
- Translate them literally into English, preserving the original sentence structure and rhythm
- Use this translation as the replacement for the flagged sentence
- The replacement must cover the same factual ground — do not change the underlying claim

If you cannot find a good foreign language source after searching:
- Rewrite the flagged sentence to break its AI-like symmetry and structure
- Add natural human messiness — interruptions, abrupt stops, incomplete thoughts, tangents
- Keep the same fact but present it differently
- Example: "His training combines explosive power with endurance work, speed work with agility exercises" becomes "The training is built around explosions. Short violent sprints. Then agility. Then strength — in that order, every time."

RULES FOR ALL REPLACEMENTS:
- Do NOT change any underlying facts or statistics
- Do NOT bold anything
- Do NOT use m-dashes
- Avoid symmetrical list structures ("X with Y, A with B")
- Vary sentence length dramatically in replacements
- ${citationNote}

After replacing all flagged sentences, return the COMPLETE rewritten draft with replacements inserted in the correct positions.
No preamble. Just the full piece.`;

  const flaggedList = flaggedSentences.slice(0, 8).map((s, i) => (i + 1) + ". " + s).join("\n");

  try {
    const humanized = await claude(system, `Text type: ${typeConfig.name}\n\nFlagged sentences to replace:\n${flaggedList}\n\nFull draft:\n${draft}`, true);
    res.json({ humanized: humanized.replace(/\*\*/g, "").replace(/\[\d+\]/g, "").trim() });
  } catch (err) {
    res.status(500).json({ error: "Humanize error: " + (err.response?.data?.error?.message || err.message) });
  }
});

app.post("/polish", async (req, res) => {
  const { text, textType, citations } = req.body;
  if (!text) return res.status(400).json({ error: "Missing text" });

  const typeConfig = TEXT_TYPES[textType] || TEXT_TYPES.oped;
  const citationNote = citations
    ? "Keep all existing in-text citations exactly as they are."
    : "Remove any in-text citations from the body. Keep the SOURCES section at the end intact.";

  const system = `You are a minimal final editor. The text has already passed AI detection at a high score. Your job is purely cosmetic — fix only the things listed below. Do NOT rewrite sentences. Do NOT improve phrasing. Do NOT smooth anything out. Every word you change risks lowering the human score.

ONLY fix these specific things:
- Named individuals the reader has not met (e.g. "Sam") — delete the name or replace with a pronoun
- Phrases referencing a website database or tool (e.g. "in our database") — cut those words only
- Numbered footnote markers like [1] [2] [3] — delete them
- Phrases like "as we mentioned", "as noted above", "here is word for word" — cut them
- Any bolded text — remove the bold formatting only, keep the words
- An incomplete ending with no conclusion — add ONE closing sentence only
- DUPLICATE SENTENCES: If two sentences within a few lines of each other say the same thing in different words — delete the weaker one entirely. Do not rewrite either. Just cut one. Example: "Income inequality describes the uneven distribution of income across society. Income inequality is the extent to which income is distributed unevenly among a population." — delete one of these.
- DUPLICATE STATISTICS: If the same statistic or fact appears twice in the text — delete the second instance entirely.
- ${citationNote}

DO NOT touch anything else. Do not rephrase. Do not restructure. Do not improve flow. Do not smooth translations.
The goal is to change as few words as possible while fixing only the above issues.
Output the full piece. No preamble.`;

  try {
    const polished = await claude(system, "Polish this " + typeConfig.name + ":\n\n" + text);
    res.json({ polished: polished.replace(/\*\*/g, "").replace(/\[\d+\]/g, "").trim() });
  } catch (err) {
    res.status(500).json({ error: "Polish error: " + (err.response?.data?.error?.message || err.message) });
  }
});

app.post("/clarify", async (req, res) => {
  const { prompt, clarificationsSoFar } = req.body;
  if (!prompt) return res.status(400).json({ error: "Missing prompt" });

  const prevAnswered = clarificationsSoFar && clarificationsSoFar.length > 0
    ? clarificationsSoFar.map(c => c.question + " -> " + c.answer).join("\n")
    : "None";

  // Only use web search if the prompt contains words suggesting a recent/current event
  const needsWebSearch = /\b(recent|latest|current|new|today|2024|2025|2026|now|ongoing|just|this year|this week)\b/i.test(prompt);

  const system = `You are a sharp, intelligent writing assistant. Your job is to read the user's prompt and any previous answers, think carefully about what they actually mean, and decide if one more clarifying question is needed.

You have TWO reasons to ask a question:

REASON 1 — TOPIC AMBIGUITY: The prompt refers to something unclear.
Examples:
- "Write about the world war" -> ask which world war
- "Write about the president's policy" -> ask which country
- "Write about the war" -> ask which conflict
- "Write about the company" -> ask which company
- After user says "neither, I meant the israel-palestine war" -> now ask something specific about THAT topic, like "Are you focusing on the humanitarian crisis, the military conflict, the political negotiations, or the historical roots of the conflict?"

REASON 2 — TOO VAGUE TO SOURCE WELL: The prompt is broad enough that a more specific angle will produce better human-written sources and a lower AI score.
Examples:
- "Why is Ronaldo so good?" -> ask which aspect: athleticism, career, mentality, or technique
- "Write about climate change" -> ask which angle: science, politics, economics, or human stories
- "Write about social media" -> ask which angle: mental health, politics, business, or youth culture
- "Write about the israel-palestine war" -> ask which angle: humanitarian impact, military conflict, political negotiations, or historical roots

CRITICAL RULES:
- After a user corrects you or gives an unexpected answer, always check if a follow-up is now needed about the NEW topic they revealed. Do not just accept and move on if the new topic is still vague.
- "all" or "everything" = valid answer, move on
- "none of those, I meant X" = update your understanding AND ask a follow-up about X if X is still vague
- Do NOT ask about word count, tone, text type, or citations
- Ask ONE question maximum per round
- Stop asking when the topic is specific enough to find targeted sources
- If the prompt appears to be a school assignment brief or rubric with multiple possible topics listed, ask the user: which topic and which specific country or community do you want to focus on? This is the most important question for an assignment brief.
- If the user has already specified a topic and location from an assignment brief, do not ask again — proceed.

Previous clarifications:
\${prevAnswered}

Return ONLY valid JSON:
{"needsClarification": true, "question": "Your question here?"}
or
{"needsClarification": false}`;

  try {
    const userMsg = "Prompt: " + prompt + (clarificationsSoFar && clarificationsSoFar.length > 0 ? "\nPrevious Q&A:\n" + clarificationsSoFar.map(c => "Q: " + c.question + "\nA: " + c.answer).join("\n") : "");
    const raw = await claude(system, userMsg, needsWebSearch);
    const parsed = JSON.parse(raw.replace(/```json|```/g, "").trim());
    res.json(parsed);
  } catch (err) {
    res.json({ needsClarification: false });
  }
});

const PORT = process.env.PORT || 3131;
app.listen(PORT, () => console.log("Humanizer server running on http://localhost:" + PORT));
