TEMPLATE_VERSION = "1.0.0"

class PromptTemplate:
    @staticmethod
    def strict_qa(context: str, question: str) ->str:
        return f"""You are a document-grounded Question Answering system.

RULES (non-negotiable):
1. Answer ONLY based on the CONTEXT provided below.
2. Do NOT use prior knowledge, training data, or assumptions.
3. If the answer cannot be found in the CONTEXT, respond with exactly: INSUFFICIENT_CONTEXT
4. Be precise and concise. Do not add information not explicitly stated in the CONTEXT.
5. Do not explain your reasoning. Just answer the question directly.

CONTEXT:
{context.strip()}

QUESTION:
{question.strip()}

ANSWER:"""

    @staticmethod
    def consistency_qa(context: str, question: str, variant_label: str) -> str:
        _ = variant_label
        return f"""You are a document-grounded Question Answering system.

RULES (non-negotiable):
1. Answer ONLY based on the CONTEXT provided below.
2. Do NOT use prior knowledge, training data, or assumptions.
3. If the answer cannot be found in the CONTEXT, respond with exactly: INSUFFICIENT_CONTEXT
4. Be precise and concise. Do not add information not explicitly stated in the CONTEXT.
5. Do not explain your reasoning. Just answer the question directly.

CONTEXT:
{context.strip()}

QUESTION:
{question.strip()}

ANSWER:"""

    @staticmethod
    def distractor_qa(context: str, question: str, distractor_type: str) -> str:
        _ = distractor_type
        
        return f"""You are a document-grounded Question Answering system.

RULES (non-negotiable):
1. Answer ONLY based on the CONTEXT provided below.
2. Do NOT use prior knowledge, training data, or assumptions.
3. If the answer cannot be found in the CONTEXT, respond with exactly: INSUFFICIENT_CONTEXT
4. Be precise and concise. Do not add information not explicitly stated in the CONTEXT.
5. Do not explain your reasoning. Just answer the question directly.

CONTEXT:
{context.strip()}

QUESTION:
{question.strip()}

ANSWER:"""

    @staticmethod
    def sensitivity_qa(context: str, question: str, perturbation: str) -> str:
        _ = perturbation
        return f"""You are a document-grounded Question Answering system.

RULES (non-negotiable):
1. Answer ONLY based on the CONTEXT provided below.
2. Do NOT use prior knowledge, training data, or assumptions.
3. If the answer cannot be found in the CONTEXT, respond with exactly: INSUFFICIENT_CONTEXT
4. Be precise and concise. Do not add information not explicitly stated in the CONTEXT.
5. Do not explain your reasoning. Just answer the question directly.

CONTEXT:
{context.strip()}

QUESTION:
{question.strip()}

ANSWER:"""

TEMPLATE_REGISTRY = {
    "strict_qa": {
        "method": PromptTemplate.strict_qa,
        "used_in": ["clean_cases", "distractor_cases"],
        "escape_hatch": "INSUFFICIENT_CONTEXT",
        "version": TEMPLATE_VERSION,
    },
    "consistency_qa": {
        "method": PromptTemplate.consistency_qa,
        "used_in": ["paraphrase_cases"],
        "escape_hatch": "INSUFFICIENT_CONTEXT",
        "version": TEMPLATE_VERSION,
    },
    "distractor_qa": {
        "method": PromptTemplate.distractor_qa,
        "used_in": ["distractor_cases"],
        "escape_hatch": "INSUFFICIENT_CONTEXT",
        "version": TEMPLATE_VERSION,
        "note": "Prompt identik dengan strict_qa — model tidak tahu konteks dimodifikasi.",
    },
    "sensitivity_qa": {
        "method": PromptTemplate.sensitivity_qa,
        "used_in": ["sensitivity_cases"],
        "escape_hatch": "INSUFFICIENT_CONTEXT",
        "version": TEMPLATE_VERSION,
    },
}

# quict audit
if __name__ == "__main__":
    print("=" * 60)
    print(f"Prompt Templates — version {TEMPLATE_VERSION}")
    print("=" * 60)

    sample_context = "FastAPI uses Pydantic for data validation and supports async out of the box."
    sample_question = "What does FastAPI use for data validation?"

    for name, meta in TEMPLATE_REGISTRY.items():
        print(f"\n{'─' * 60}")
        print(f"Template  : {name}")
        print(f"Used in   : {', '.join(meta['used_in'])}")
        print(f"Version   : {meta['version']}")
        if "note" in meta:
            print(f"Note      : {meta['note']}")
            
        print(f"\n--- Preview ---")
        if name == "strict_qa":
            preview = PromptTemplate.strict_qa(sample_context, sample_question)
        elif name == "consistency_qa":
            preview = PromptTemplate.consistency_qa(
                sample_context, sample_question, "original"
            )
        elif name == "distractor_qa":
            preview = PromptTemplate.distractor_qa(
                sample_context, sample_question, "value_swap"
            )
        elif name == "sensitivity_qa":
            preview = PromptTemplate.sensitivity_qa(
                sample_context, sample_question, "question_wording"
            )

        print(preview)

    print("\n" + "=" * 60)
    print(f"Total templates: {len(TEMPLATE_REGISTRY)}")
    print("Semua template verified.")
    print("=" * 60)