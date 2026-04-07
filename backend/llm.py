import os
import json
import re
import random

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_STUDY_COACH_PROMPT = """You are KTU Assistant, a reliable academic chatbot for Kerala Technological University students.
Your job is to help students learn from their syllabus, notes, and available study material.

Core rules:
- Use the provided context as the primary source whenever it is relevant.
- If the context is weak, missing, or unrelated, say that clearly before giving general guidance.
- Do not invent syllabus facts, module numbers, marks, dates, or source titles.
- If the student only greets you or sends a vague opener, respond warmly and invite them to ask a study question.
- If the question is vague but includes a topic, ask one friendly focused follow-up question or answer with clearly stated assumptions.
- For notes, explanations, and learning plans, enforce Subject -> Module -> Action. For mock tests and flashcards, module is optional because PYQs are topic/subject-based.
- Do not explain a subject when the student only mentions the subject name.
- Do not talk about internal/admin document management. Avoid saying the student has uploaded PDFs or files.
- Never mention prompts, system instructions, "study context", "provided context", or "the student's question" in the answer.
- For casual, non-study, or setup messages, keep the reply to one short sentence unless the student asks for detail.
- Keep answers student-friendly, structured, and exam-aware.
- When context includes source links, mention the most relevant source names naturally when helpful.
- For math, algorithms, code, and derivations, show the steps instead of only the final answer.
- When the question is about PYQs or past exam questions, answer concisely and stay directly grounded in the PYQ text.
- If the student asks for cheating, bypassing exams, or non-academic misuse, refuse briefly and redirect to learning help."""

SYSTEM_PROMPTS = {
    "teacher": f"""{BASE_STUDY_COACH_PROMPT}

Mode: Teacher
- Start with the simplest useful explanation.
- Break the answer into short steps.
- Include a small example or analogy when it helps.
- End with a quick recap if the topic is complex.""",
    "exam": f"""{BASE_STUDY_COACH_PROMPT}

Mode: Exam
- Write scoring-point answers suitable for KTU exams.
- Prefer headings and compact bullet points.
- Include definitions, formulas, diagrams-to-draw notes, and key differences when relevant.
- Avoid long storytelling unless the student asks for detailed explanation.""",
    "revision": f"""{BASE_STUDY_COACH_PROMPT}

Mode: Revision
- Give a quick revision sheet.
- Prioritize key terms, formulas, memory hooks, and likely mistakes.
- Keep it compact and easy to scan.
- Add 2-4 self-check questions when useful.""",
}

SUBJECT_ALIASES = {
    "aad": "Algorithm Analysis and Design",
    "algorithm analysis": "Algorithm Analysis and Design",
    "algorithm analysis and design": "Algorithm Analysis and Design",
    "cd": "Compiler Design",
    "compiler design": "Compiler Design",
    "cgip": "Computer Graphics and Image Processing",
    "computer graphics": "Computer Graphics and Image Processing",
    "computer graphics and image processing": "Computer Graphics and Image Processing",
    "ieft": "Industrial Economics and Foreign Trade",
    "industrial economics": "Industrial Economics and Foreign Trade",
    "industrial economics and foreign trade": "Industrial Economics and Foreign Trade",
    "python": "Programming in Python",
    "programming in python": "Programming in Python",
}
SUBJECT_SHORT_NAMES = {
    "Algorithm Analysis and Design": "AAD",
    "Computer Graphics and Image Processing": "CGIP",
    "Industrial Economics and Foreign Trade": "IEFT",
}

QUESTION_INTENT_WORDS = {
    "analyse",
    "analyze",
    "answer",
    "compare",
    "define",
    "derive",
    "describe",
    "differentiate",
    "explain",
    "find",
    "give",
    "how",
    "list",
    "notes",
    "prove",
    "short",
    "summarize",
    "what",
    "when",
    "where",
    "why",
    "write",
}
ACTION_ALIASES = {
    "mock_test": {"mock", "test", "mocktest", "quiz", "questions", "practice"},
    "flashcards": {"flash", "flashcard", "flashcards", "cards"},
    "notes": {"note", "notes", "explain", "explanation", "overview", "summary", "teach", "learn"},
    "learning_plan": {"plan", "schedule", "roadmap", "learn"},
}
SESSION_MEMORY: dict[str, dict[str, str | int | None]] = {}

SHORT_REPLY_OPENERS = {
    "lets play a game": "Yes, let's play a fun game. I'll ask you questions based on your syllabus and notes.",
    "let's play a game": "Yes, let's play a fun game. I'll ask you questions based on your syllabus and notes.",
    "play a game": "Yes, let's play a fun game. I'll ask you questions based on your syllabus and notes.",
    "i want to play a game": "Yes, let's play a fun game. I'll ask you questions based on your syllabus and notes.",
    "a fun game": "Perfect. I'll ask you quick questions from your syllabus and notes.",
    "fun game": "Perfect. I'll ask you quick questions from your syllabus and notes.",
    "quick fun game": "Perfect. I'll ask you quick questions from your syllabus and notes.",
}


def _is_greeting(prompt: str) -> bool:
    normalized = " ".join((prompt or "").strip().lower().split())
    if not normalized:
        return False

    greeting_tokens = {
        "hi",
        "hii",
        "hiii",
        "hey",
        "hello",
        "yo",
        "sup",
        "hola",
        "namaste",
        "good morning",
        "good afternoon",
        "good evening",
    }
    assistant_names = {"kai", "ktu assistant", "assistant", "bot"}
    polite_fillers = {"there", "dear"}

    if normalized in greeting_tokens:
        return True

    stripped = normalized.replace(",", "").replace("!", "").replace(".", "")
    words = stripped.split()
    if not words:
        return False

    if len(words) <= 3 and words[0] in greeting_tokens:
        remaining = " ".join(word for word in words[1:] if word not in polite_fillers)
        return not remaining or remaining in assistant_names

    for greeting in ("good morning", "good afternoon", "good evening"):
        if stripped.startswith(f"{greeting} ") and len(words) <= 4:
            remaining = " ".join(
                word for word in words[len(greeting.split()):] if word not in polite_fillers
            )
            return not remaining or remaining in assistant_names

    return False


def _greeting_reply(student_name: str | None = None) -> str:
    if student_name:
        return f"Hey {student_name}, tell me the subject or topic you want to work on."
    return "Hey, tell me the subject or topic you want to work on."


def _is_casual_opener(prompt: str) -> bool:
    normalized = " ".join((prompt or "").strip().lower().split())
    if not normalized:
        return False

    casual_openers = {
        "what are you doing",
        "what are you doing?",
        "what do you do",
        "what do you do?",
        "who are you",
        "who are you?",
        "how are you",
        "how are you?",
        "how can you help",
        "how can you help?",
        "what can you do",
        "what can you do?",
    }
    return normalized in casual_openers


def _casual_opener_reply(student_name: str | None = None) -> str:
    name_part = f", {student_name}" if student_name else ""
    return (
        f"I am here with you{name_part}. "
        "We can pick a subject, revise a module, or solve a specific doubt."
    )


def _short_reply(prompt: str) -> str | None:
    normalized = _normalize_text(prompt).rstrip(".!?")
    return SHORT_REPLY_OPENERS.get(normalized)


def _normalize_text(prompt: str) -> str:
    return " ".join((prompt or "").strip().lower().split())


def _plain_text(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize_text(prompt)).strip()


def _subject_label(subject: str) -> str:
    short_name = SUBJECT_SHORT_NAMES.get(subject)
    return f"{subject} ({short_name})" if short_name else subject


def _normalize_module_token(value: str) -> int:
    roman_modules = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
    value = value.lower()
    return roman_modules.get(value, int(value))


def _detect_subject(prompt: str) -> str | None:
    plain = _plain_text(prompt)
    for alias, subject in sorted(SUBJECT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        alias_pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
        if re.search(alias_pattern, plain):
            return subject
    return None


def _detect_module_number(prompt: str) -> int | None:
    plain = _plain_text(prompt)
    match = re.search(r"\b(?:module|mod|m)\s*([1-5]|i{1,3}|iv|v)\b", plain)
    if not match:
        return None
    return _normalize_module_token(match.group(1))


def _detect_action(prompt: str) -> str | None:
    words = set(_plain_text(prompt).split())
    for action, aliases in ACTION_ALIASES.items():
        if words & aliases:
            return action
    return None


def _module_action_reply(subject: str, module_number: int) -> str:
    return f"What would you like for {_subject_label(subject)} Module {module_number}: notes, flashcards, mock test, or learning plan?"


def _missing_module_reply(subject: str | None = None) -> str:
    if subject:
        return f"Which module in {_subject_label(subject)} do you want?"
    return "Please specify the subject and module number to generate accurate content."


def _session_key(value: str | None = None) -> str:
    return value or "__anonymous__"


def _get_session_memory(session_key: str) -> dict[str, str | int | None]:
    return SESSION_MEMORY.setdefault(
        session_key,
        {"current_subject": None, "current_module": None, "current_action": None},
    )


def _update_session_memory(
    session_key: str,
    subject: str | None = None,
    module_number: int | None = None,
    action: str | None = None,
) -> dict[str, str | int | None]:
    memory = _get_session_memory(session_key)
    if subject:
        if memory.get("current_subject") != subject:
            memory["current_module"] = None
            memory["current_action"] = None
        memory["current_subject"] = subject
    if module_number is not None:
        memory["current_module"] = module_number
    if action:
        memory["current_action"] = action
    return memory


def _detect_subject_opener(prompt: str) -> tuple[str, str | None] | None:
    normalized = _normalize_text(prompt)
    if not normalized:
        return None

    plain = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    words = set(plain.split())

    if words & QUESTION_INTENT_WORDS:
        return None

    matched_subject = None
    for alias, subject in sorted(SUBJECT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        alias_pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
        if re.search(alias_pattern, plain):
            matched_subject = subject
            break

    if not matched_subject:
        return None

    module_match = re.search(r"\b(?:module|mod)\s*([1-5]|i{1,3}|iv|v)\b", plain)
    module = module_match.group(1).upper() if module_match else None
    return matched_subject, module


def _subject_opener_reply(subject: str, module: str | None = None) -> str:
    if module:
        return (
            f"Great, let's learn {subject} Module {module}. "
            "Should I start with a quick overview, or do you have a topic in mind?"
        )

    return (
        f"Great, let's learn {subject}. "
        "Which module should we open first?"
    )


def _remote_llm_enabled() -> bool:
    value = os.getenv("ENABLE_REMOTE_LLM")
    if value is None:
        return bool(os.getenv("GROQ_API_KEY"))
    return value.lower() == "true"


def _normalize_mode(mode: str | None) -> str:
    normalized = (mode or "").strip().lower()
    if normalized in SYSTEM_PROMPTS:
        return normalized
    return "teacher"


def _get_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=8.0,
        max_retries=0,
    )


def _fallback_reply(prompt: str, context: str) -> str:
    clean_prompt = prompt.strip()
    if context and "disabled" not in context.lower() and "unavailable" not in context.lower() and "no documents" not in context.lower():
        return f"I couldn't reach the live AI service, but I found this in the study material:\n\n{context[:1200]}"

    return (
        "I couldn't reach the live AI service right now, but your server is working again. "
        f"For now, I can only give a basic fallback response to: \"{clean_prompt}\". "
        "Try again in a moment, or ask a more specific subject or module question."
    )


def _context_available(context: str) -> bool:
    lowered = (context or "").lower()
    unavailable_markers = (
        "disabled",
        "unavailable",
        "no documents",
        "no study material",
        "no module",
        "could not extract",
    )
    return bool(context and not any(marker in lowered for marker in unavailable_markers))


def _context_sentences(context: str, limit: int = 40) -> list[str]:
    cleaned = re.sub(r"Source: \[[^\]]+\]\([^)]+\)", " ", context)
    cleaned = re.sub(r"\s+", " ", cleaned)
    parts = re.split(r"(?<=[.!?])\s+| o |\uf0b7|\uf0a7|•", cleaned)
    sentences = []
    seen = set()
    for part in parts:
        sentence = part.strip(" -:;")
        if len(sentence) < 35 or len(sentence) > 220:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(sentence)
        if len(sentences) >= limit:
            break
    return sentences


def _scope_label(module_number: int | None) -> str:
    return f"Module {module_number}" if module_number is not None else "PYQ-focused subject practice"


def _fallback_mock_test(subject: str, module_number: int | None, context: str, regenerate: bool = False) -> str:
    if not _context_available(context):
        return f"I could not find enough exam material for {subject} to generate a grounded mock test."

    sentences = _context_sentences(context, limit=80)
    if regenerate:
        random.shuffle(sentences)

    scope = _scope_label(module_number)
    questions = []
    for sentence in sentences:
        if len(questions) >= 15:
            break
        difficulty = "Medium"
        if any(term in sentence.lower() for term in ("define", "definition", "what is", "properties")):
            difficulty = "Easy"
        elif any(term in sentence.lower() for term in ("algorithm", "complexity", "derive", "analysis")):
            difficulty = "Hard"
        questions.append(
            f"{len(questions) + 1}. [{difficulty}] Short answer: Explain this {scope} point: {sentence}"
        )

    while len(questions) < 15:
        if sentences:
            sentence = sentences[len(questions) % len(sentences)]
            questions.append(
                f"{len(questions) + 1}. [Medium] Short answer: Explain this {scope} point: {sentence}"
            )
            continue
        questions.append(
            f"{len(questions) + 1}. [Medium] Short answer: Write an exam-relevant note from {subject}."
        )

    return (
        f"10-Minute Mock Test: {subject} - {scope}\n"
        "Answer all 15 questions. PYQ patterns are prioritized. Keep answers brief and exam-focused.\n\n"
        + "\n".join(questions)
    )


def _fallback_flashcards(subject: str, module_number: int | None, context: str, num_cards: int, regenerate: bool = False) -> str:
    if not _context_available(context):
        return f"I could not find enough exam material for {subject} to generate grounded flashcards."

    sentences = _context_sentences(context, limit=max(num_cards * 2, 20))
    if not sentences:
        return f"I could not find enough exam material for {subject} to generate grounded flashcards."
    if regenerate:
        random.shuffle(sentences)

    scope = _scope_label(module_number)
    cards = []
    for sentence in sentences:
        if len(cards) >= num_cards:
            break
        term = sentence.split(" is ", 1)[0] if " is " in sentence else f"{subject} {scope}"
        cards.append(
            f"{len(cards) + 1}. Q: {term.strip()}\nA: {sentence}"
        )

    return "\n\n".join(cards)


def _build_grounded_generation_prompt(kind: str, subject: str, module_number: int | None, context: str, num_cards: int | None = None) -> str:
    scope = f"Module {module_number}" if module_number is not None else "the subject as a whole"
    if kind == "mock_test":
        return f"""Generate a 10-minute mock test for {subject}, {scope}.

Study material:
{context}

Rules:
- Use only the study material above. Do not add outside facts.
- Exactly 15 questions.
- Moderate difficulty, exam-oriented.
- Prioritize PYQ knowledge over notes whenever PYQ knowledge is available.
- Questions should be inspired by or derived from PYQs, repeated concepts, and common exam patterns.
- Prioritize high-weightage-looking concepts, repeated terms, definitions, formulas, algorithms, properties, and applications.
- Avoid redundancy and cover different parts of the requested scope.
- Include difficulty tags: [Easy], [Medium], or [Hard].
- Use mostly MCQs, with short-answer questions only when better for the content.
- For MCQs, include four options and mark the answer.
- Do not mention prompts, context, PDFs, files, or uploads."""

    return f"""Generate {num_cards} flashcards for {subject}, {scope}.

Study material:
{context}

Rules:
- Use only the study material above. Do not add outside facts.
- Prioritize PYQ-derived key terms and commonly tested concepts over notes when PYQ knowledge is available.
- Use Q/A or Term/Definition format.
- Keep each card concise and memory-focused.
- Cover key concepts, definitions, formulas, critical facts, and exam-relevant points.
- Avoid unnecessary explanation and redundancy.
- Do not mention prompts, context, PDFs, files, or uploads."""


def generate_mock_test_from_context(subject: str, module_number: int | None, context: str, regenerate: bool = False) -> str:
    if not _context_available(context):
        return f"I could not find enough exam material for {subject} to generate a grounded mock test."
    if not _remote_llm_enabled():
        return _fallback_mock_test(subject, module_number, context, regenerate=regenerate)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You generate concise exam practice strictly from supplied study material."},
                {"role": "user", "content": _build_grounded_generation_prompt("mock_test", subject, module_number, context)},
            ],
            temperature=0.6 if regenerate else 0.3,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return _fallback_mock_test(subject, module_number, context, regenerate=regenerate)


def _fallback_structured_mock_test(
    subject: str,
    module_number: int,
    context: str,
    num_questions: int = 15,
    regenerate: bool = False,
) -> list[dict]:
    lowered = context.lower()
    templates = [
        (
            {"merge sort", "mergesort"},
            {
                "question": "What is the worst-case time complexity of merge sort?",
                "options": {"A": "O(n)", "B": "O(n log n)", "C": "O(log n)", "D": "O(n^2)"},
                "correct_answer": "B",
            },
        ),
        (
            {"binary search"},
            {
                "question": "Which condition is essential for applying binary search correctly?",
                "options": {
                    "A": "The input must be sorted",
                    "B": "The input must be stored as a linked list",
                    "C": "The input must contain only distinct prime numbers",
                    "D": "The input must be traversed sequentially from the first element",
                },
                "correct_answer": "A",
            },
        ),
        (
            {"big o", "big-o", "upper bound"},
            {
                "question": "What does Big-O notation primarily describe in algorithm analysis?",
                "options": {
                    "A": "A lower bound on running time",
                    "B": "An upper bound on growth rate",
                    "C": "The exact execution time on a processor",
                    "D": "The memory address of each operation",
                },
                "correct_answer": "B",
            },
        ),
        (
            {"omega", "lower bound"},
            {
                "question": "Which asymptotic notation is used to express a lower bound?",
                "options": {"A": "Big-O", "B": "Theta", "C": "Omega", "D": "Little-o"},
                "correct_answer": "C",
            },
        ),
        (
            {"theta", "tight bound"},
            {
                "question": "What does Theta notation represent?",
                "options": {
                    "A": "Only the worst-case input size",
                    "B": "A tight asymptotic bound",
                    "C": "Only the lower bound of an algorithm",
                    "D": "The number of recursive calls ignored in analysis",
                },
                "correct_answer": "B",
            },
        ),
        (
            {"dijkstra"},
            {
                "question": "Which strategy is used by Dijkstra's shortest path algorithm?",
                "options": {
                    "A": "Divide and conquer",
                    "B": "Greedy selection with edge relaxation",
                    "C": "Backtracking over all possible paths",
                    "D": "Depth-first traversal without weights",
                },
                "correct_answer": "B",
            },
        ),
        (
            {"avl"},
            {
                "question": "Why are rotations performed in an AVL tree?",
                "options": {
                    "A": "To maintain height balance after updates",
                    "B": "To convert the tree into a graph",
                    "C": "To remove all leaf nodes",
                    "D": "To sort keys using hashing",
                },
                "correct_answer": "A",
            },
        ),
        (
            {"topological"},
            {
                "question": "For which type of graph is topological ordering defined?",
                "options": {
                    "A": "Undirected cyclic graph",
                    "B": "Directed acyclic graph",
                    "C": "Complete weighted graph",
                    "D": "Disconnected undirected graph only",
                },
                "correct_answer": "B",
            },
        ),
        (
            {"divide and conquer"},
            {
                "question": "What is the central idea of the divide-and-conquer technique?",
                "options": {
                    "A": "Solve a problem by splitting it into smaller subproblems and combining results",
                    "B": "Check all possible solutions one by one",
                    "C": "Always choose the locally best option",
                    "D": "Store every previous answer in a table",
                },
                "correct_answer": "A",
            },
        ),
    ]

    questions = []
    for triggers, question in templates:
        if any(trigger in lowered for trigger in triggers):
            questions.append(question)
        if len(questions) >= num_questions:
            break
    if regenerate:
        random.shuffle(questions)
    return questions[:num_questions]


def _structured_mock_prompt(subject: str, module_number: int, context: str, num_questions: int) -> str:
    return f"""You are an expert KTU exam paper setter.

Generate a high-quality mock test for the selected module of {subject} using the provided academic context.

Academic context:
{context}

Return ONLY strict JSON in this exact format:
[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "correct_answer": "A"
  }}
]

Rules:
- Exactly {num_questions} questions.
- Use ONLY the academic context. If context is insufficient for a question, skip it instead of guessing.
- Generate unique, well-structured MCQs. Do not copy or repeat lines directly from the context.
- Each question must be clear, exam-oriented, concept-based, and aligned with KTU exam standards.
- Cover different topics within the selected module; do not repeat the same idea in different wording.
- Mix easy definitions, medium concept-understanding, and hard application-based questions.
- Each MCQ must have one clear question, four distinct options A-D, and only one correct answer letter.
- Distractors must be plausible, not copied from unrelated lines, and not all look similar.
- Do not repeat "Module {module_number}" or similar module labels inside each question.
- Do not use vague stems like "Which option best represents..." or options like "A concept used in...".
- Use natural university-style stems, for example "What is the worst-case time complexity of merge sort?"
- Do not include explanations, source hints, markdown, prose, or any text outside the JSON array."""


def _is_quality_mock_question(item: dict, module_number: int) -> bool:
    question = str(item.get("question", "")).strip()
    raw_options = item.get("options", {})
    option_values = raw_options.values() if isinstance(raw_options, dict) else raw_options
    combined_options = " ".join(str(option) for option in option_values)
    lowered = f"{question} {combined_options}".lower()
    banned_fragments = (
        f"module {module_number}",
        "which option best represents",
        "which statement best matches",
        "a concept used in",
        "a topic mainly concerned with",
        "a technique primarily related to",
        "a result associated with",
    )
    if any(fragment in lowered for fragment in banned_fragments):
        return False
    if len(question.split()) < 6:
        return False
    return len(set(str(option).strip().lower() for option in option_values)) == 4


def generate_structured_mock_test_from_context(
    subject: str,
    module_number: int,
    context: str,
    num_questions: int = 15,
    regenerate: bool = False,
) -> list[dict]:
    if not _context_available(context):
        return _fallback_structured_mock_test(subject, module_number, context, num_questions, regenerate=regenerate)
    if not _remote_llm_enabled():
        return _fallback_structured_mock_test(subject, module_number, context, num_questions, regenerate=regenerate)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You generate strict JSON MCQ tests only from supplied academic resources.",
                },
                {
                    "role": "user",
                    "content": _structured_mock_prompt(subject, module_number, context, num_questions),
                },
            ],
            temperature=0.55 if regenerate else 0.25,
        )
        content = (response.choices[0].message.content or "[]").strip()
        if not content.startswith("["):
            match = re.search(r"\[[\s\S]*\]", content)
            content = match.group(0) if match else content
        payload = json.loads(content)
        questions = payload.get("questions", []) if isinstance(payload, dict) else payload
        cleaned = []
        for item in questions:
            raw_options = item.get("options", {})
            if isinstance(raw_options, list) and len(raw_options) == 4:
                raw_options = dict(zip(["A", "B", "C", "D"], raw_options))
            options = {
                label: str(raw_options.get(label, "")).strip()
                for label in ("A", "B", "C", "D")
            }
            correct_answer = str(item.get("correct_answer", "")).strip().upper()
            question = str(item.get("question", "")).strip()
            if not question or correct_answer not in options or not all(options.values()):
                continue
            normalized_item = {"question": question, "options": options, "correct_answer": correct_answer}
            if not _is_quality_mock_question(normalized_item, module_number):
                continue
            cleaned.append(normalized_item)
            if len(cleaned) >= num_questions:
                break
        if len(cleaned) == num_questions:
            return cleaned
    except Exception:
        pass

    return _fallback_structured_mock_test(subject, module_number, context, num_questions, regenerate=regenerate)


def generate_flashcards_from_context(
    subject: str,
    module_number: int | None,
    context: str,
    num_cards: int = 20,
    regenerate: bool = False,
) -> str:
    if not _context_available(context):
        return f"I could not find enough exam material for {subject} to generate grounded flashcards."
    if not _remote_llm_enabled():
        return _fallback_flashcards(subject, module_number, context, num_cards, regenerate=regenerate)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You generate concise flashcards strictly from supplied study material."},
                {
                    "role": "user",
                    "content": _build_grounded_generation_prompt(
                        "flashcards", subject, module_number, context, num_cards=num_cards
                    ),
                },
            ],
            temperature=0.6 if regenerate else 0.3,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return _fallback_flashcards(subject, module_number, context, num_cards, regenerate=regenerate)


def _fallback_module_notes(subject: str, module_number: int, context: str) -> str:
    if not _context_available(context):
        return f"I could not find enough Module {module_number} material for {_subject_label(subject)}."

    sentences = _context_sentences(context, limit=6)
    if not sentences:
        return f"I could not find enough Module {module_number} material for {_subject_label(subject)}."

    points = "\n".join(f"- {sentence}" for sentence in sentences[:6])
    return f"{_subject_label(subject)} Module {module_number} notes:\n{points}"


def generate_module_notes_from_context(subject: str, module_number: int, context: str) -> str:
    if not _context_available(context):
        return f"I could not find enough Module {module_number} material for {_subject_label(subject)}."
    if not _remote_llm_enabled():
        return _fallback_module_notes(subject, module_number, context)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Give concise module notes strictly from supplied study material. Do not mention prompts, context, PDFs, or files.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Create concise notes for {subject} Module {module_number} using only this study material.\n\n"
                        f"{context}\n\n"
                        "Keep it brief, exam-focused, and use bullets."
                    ),
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return _fallback_module_notes(subject, module_number, context)


def _fallback_learning_plan(subject: str, module_number: int, context: str) -> str:
    if not _context_available(context):
        return "No content found for this module in the selected subject."

    sentences = _context_sentences(context, limit=5)
    if not sentences:
        return "No content found for this module in the selected subject."

    steps = "\n".join(f"{index + 1}. {sentence}" for index, sentence in enumerate(sentences[:5]))
    return f"Learning plan for {_subject_label(subject)} Module {module_number}:\n{steps}"


def generate_learning_plan_from_context(subject: str, module_number: int, context: str) -> str:
    if not _context_available(context):
        return "No content found for this module in the selected subject."
    if not _remote_llm_enabled():
        return _fallback_learning_plan(subject, module_number, context)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Create concise learning plans strictly from supplied study material. Do not mention prompts, context, PDFs, or files.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Create a concise learning plan for {subject} Module {module_number} using only this study material.\n\n"
                        f"{context}\n\nUse numbered steps only."
                    ),
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return _fallback_learning_plan(subject, module_number, context)


def _generate_controlled_action(subject: str, module_number: int | None, action: str) -> str:
    from .pyq_engine import query_pyq_context
    from .rag_engine import query_module_context, query_subject_notes_context

    if action in {"mock_test", "flashcards"}:
        pyq_context = query_pyq_context(subject, k=12)
        if module_number is not None:
            notes_context = query_module_context(subject, module_number, k=8)
        else:
            notes_context = query_subject_notes_context(subject, query=action.replace("_", " "), k=6)
        context = f"PYQ knowledge (high priority):\n{pyq_context}\n\nNotes knowledge:\n{notes_context}"
    elif module_number is not None:
        context = query_module_context(subject, module_number, k=10)
    else:
        return _missing_module_reply(subject)

    if not _context_available(context):
        return "No content found for the selected subject and scope."
    if action == "mock_test":
        return generate_mock_test_from_context(subject, module_number, context)
    if action == "flashcards":
        return generate_flashcards_from_context(subject, module_number, context, num_cards=20)
    if action == "notes":
        return generate_module_notes_from_context(subject, module_number, context)
    if action == "learning_plan":
        return generate_learning_plan_from_context(subject, module_number, context)
    return _module_action_reply(subject, module_number)


def _controlled_study_reply(prompt: str, session_key: str) -> str | None:
    memory = _get_session_memory(session_key)
    subject = _detect_subject(prompt)
    module_number = _detect_module_number(prompt)
    action = _detect_action(prompt)
    current_subject = memory.get("current_subject")
    current_module = memory.get("current_module")

    if not subject and action in {"mock_test", "flashcards"} and current_subject:
        subject = str(current_subject)
        module_number = int(current_module) if current_module else None
        _update_session_memory(session_key, action=action)
        return _generate_controlled_action(subject, module_number, action)

    if not subject and action and current_subject and current_module:
        subject = str(current_subject)
        module_number = int(current_module)
        _update_session_memory(session_key, action=action)
        return _generate_controlled_action(subject, module_number, action)

    if not subject and module_number is not None and current_subject:
        subject = str(current_subject)
    elif not subject and module_number is not None:
        return "Please specify the subject."

    if not subject and action:
        if action in {"mock_test", "flashcards"}:
            return "Please specify the subject."
        return "Please specify the subject and module number to generate accurate content."

    if not subject:
        return None

    if module_number is None:
        _update_session_memory(session_key, subject=subject, action=action)
        if action in {"mock_test", "flashcards"}:
            return _generate_controlled_action(subject, None, action)
        return _missing_module_reply(subject)

    _update_session_memory(session_key, subject=subject, module_number=module_number, action=action)

    if action is None:
        return _module_action_reply(subject, module_number)

    return _generate_controlled_action(subject, module_number, action)


def _build_user_prompt(prompt: str, context: str) -> str:
    return f"""Use the study context below to answer the student's question.

Study context:
{context}

Student question:
{prompt}

Answer requirements:
- If the study context answers the question, rely on it.
- If only part of the answer is in the context, separate context-based points from general guidance.
- If the context does not answer the question, say so briefly and then help from general academic knowledge.
- Do not say the student has PDFs, files, uploads, or documents. Treat the context as your study knowledge.
- Do not mention prompts, instructions, context, provided material, or "the student's question".
- If the material is PYQ-focused, keep the answer concise and cite only points directly supported by the PYQ text.
- For casual, non-study, or setup messages, reply in one short sentence and ask at most one question.
- Keep the answer in the selected response mode."""


def ask_llm(
    prompt: str,
    mode: str = "teacher",
    student_name: str | None = None,
    session_key: str | None = None,
) -> str:
    session_key = _session_key(session_key or student_name)
    if _is_greeting(prompt):
        return _greeting_reply(student_name)
    if _is_casual_opener(prompt):
        return _casual_opener_reply(student_name)
    short_reply = _short_reply(prompt)
    if short_reply:
        return short_reply
    controlled_reply = _controlled_study_reply(prompt, session_key)
    if controlled_reply:
        return controlled_reply
    subject_opener = _detect_subject_opener(prompt)
    if subject_opener:
        subject, module = subject_opener
        return _subject_opener_reply(subject, module)

    # Get relevant context from the study material.
    from .pyq_engine import query_pyq_context
    from .rag_engine import query_rag

    try:
        prompt_plain = _plain_text(prompt)
        if re.search(r"\b(pyq|previous year|question paper|past question)\b", prompt_plain):
            detected_subject = _detect_subject(prompt)
            if detected_subject:
                context = query_pyq_context(detected_subject, prompt, k=8)
            else:
                context = "Please specify the subject to search PYQs."
        else:
            context = query_rag(prompt, doc_type="notes")
    except Exception:
        context = "Document search is currently unavailable."

    if not _remote_llm_enabled():
        return _fallback_reply(prompt, context)

    normalized_mode = _normalize_mode(mode)
    system_prompt = SYSTEM_PROMPTS[normalized_mode]

    full_prompt = _build_user_prompt(prompt, context)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return _fallback_reply(prompt, context)
