import anthropic

SYSTEM_PROMPT = """You are an expert prompt engineer. When given raw text input, you will:
1. Analyze the text to understand its intent and context
2. Identify improvements and additions needed
3. Refine and rewrite the prompt for maximum clarity and effectiveness

Output ONLY two optimized prompts with no commentary, analysis, or explanation.
Format your response exactly as:

CLAUDE OPTIMIZED PROMPT:
[optimized prompt for Claude]

GPT-5X OPTIMIZED PROMPT:
[optimized prompt for GPT-5X]

Claude prompts should leverage: constitutional AI alignment, nuanced instruction-following, multi-turn context, XML tags for structure, and explicit reasoning requests.
GPT-5X prompts should leverage: system/user role separation, chain-of-thought triggers, precise formatting directives, and explicit output structure requirements."""


def generate_optimized_prompts(raw_text: str) -> str:
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": raw_text,
            }
        ],
    )

    for block in response.content:
        if block.type == "text":
            return block.text

    return ""


def main():
    print("Prompt Generator Agent")
    print("=" * 40)
    print("Enter your raw text (press Enter twice when done):")
    print()

    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    raw_text = "\n".join(lines).strip()

    if not raw_text:
        print("No input provided.")
        return

    print("\nGenerating optimized prompts...\n")
    result = generate_optimized_prompts(raw_text)
    print(result)


if __name__ == "__main__":
    main()
