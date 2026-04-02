"""Prompt templates for generation, evaluation, and annotation.

All prompts used across the system are defined here for consistency.
"""

# --- RAG Generation Prompts ---

STANDARD_RAG_PROMPT = """Answer the following question based on the provided context passages.

Context:
{context}

Question: {question}

Answer:"""

SCATTER_AWARE_SYNTHESIS_PROMPT = """The following passages contain scattered information about {entity} from different parts of a document. Synthesize a complete description that covers all mentioned attributes.

Passages:
{context}

Question: {question}

Provide a comprehensive answer that integrates information from all passages:"""

# --- Query Classification Prompts ---

QUERY_CLASSIFICATION_PROMPT = """Classify this question about a document as "localized" or "scattered".

Rules:
- "localized": The answer can be found in 1-2 consecutive paragraphs of the document.
- "scattered": The answer requires combining information from 3+ distant parts of the document.

Here are examples:

Question: "What year was the company founded?"
{{"query_type": "localized", "focus_type": "event", "gold_entity": null}}
Reasoning: Founding year is typically stated once in one place.

Question: "What is the penalty for early termination?"
{{"query_type": "localized", "focus_type": "event", "gold_entity": null}}
Reasoning: Contract penalty terms are usually in a single clause.

Question: "Describe the main character's personality and how it changes throughout the story."
{{"query_type": "scattered", "focus_type": "entity", "gold_entity": "main character"}}
Reasoning: Personality traits and character development are revealed across many scenes.

Question: "What are all the responsibilities of the licensee under this agreement?"
{{"query_type": "scattered", "focus_type": "entity", "gold_entity": "licensee"}}
Reasoning: Responsibilities are typically spread across multiple sections of a contract.

Now classify this question:

Question: "{question}"

Respond with ONLY a JSON object:
{{
  "query_type": "localized" or "scattered",
  "focus_type": "entity" or "event" or "relation",
  "gold_entity": "name of the main entity" or null
}}"""

# --- ICS Attribute Presence Checker ---

ICS_PRESENCE_CHECK_PROMPT = """You are checking whether a specific attribute is mentioned in an answer.

Answer text:
"{generated_answer}"

Attribute to check: {attribute_name}
Gold evidence: {gold_evidence}

Respond with ONLY "yes" or "no"."""

# --- LLM-as-Judge Prompt ---

LLM_JUDGE_PROMPT = """Rate the following answer on three dimensions. The answer was generated for the given question based on retrieved context passages.

Question: {question}

Reference answer: {reference}

Generated answer: {answer}

Rate each dimension from 1-5:

1. **Completeness** (1=missing most info, 5=covers all key points from reference)
2. **Accuracy** (1=major errors, 5=factually consistent with reference)
3. **Coherence** (1=disorganized/contradictory, 5=well-structured and clear)

Respond with ONLY a JSON object:
{{"completeness": <1-5>, "accuracy": <1-5>, "coherence": <1-5>}}"""

# --- Entity Annotation Prompts (for ScatterQA benchmark) ---

PARAGRAPH_ATTRIBUTE_CLASSIFIER = """What type of information does this paragraph provide about the character "{entity}"?

Paragraph:
"{paragraph}"

Choose ONE category:
- appearance: physical description, clothing, distinguishing features
- background: origin, upbringing, education, past events
- personality: temperament, habits, behavioral patterns
- relationships: connections to other characters
- arc: character development, changes throughout the story
- none: paragraph does not contain meaningful information about this character

Respond with ONLY the category name."""

QA_GENERATION_PROMPT = """Based on these annotated passages about the character "{entity}" from a novel, generate a question and gold answer.

Question type: {question_type}

Annotated passages:
{passages}

Generate a JSON object:
{{"question": "...", "gold_answer": "..."}}

The question should specifically ask about the character's {question_type}.
The gold answer should synthesize information from ALL provided passages."""

# --- Scatter Category Classification ---

SCATTER_CATEGORY_PROMPT = """Classify the scatter pattern for this query about entity "{entity}".

Query: "{question}"
Number of relevant chunks: {num_chunks}
Chunk positions (as fraction of document): {positions}

Categories:
- progressive_accumulation: Info builds gradually, each mention adds a little
- distributed_attributes: Different attributes in different sections
- contradictory_evolution: Entity description changes over time
- cross_reference: Understanding entity A requires entity B's info elsewhere
- implicit_scatter: Entity not named directly, described via pronouns/references

Respond with ONLY the category name."""
