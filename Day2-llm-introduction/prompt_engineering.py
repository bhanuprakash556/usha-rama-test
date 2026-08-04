# Prompt Engineering Techniques – Simple Examples
# Use case: Campus Cafe — reply to customer reviews
#
# Copy each CELL into a Jupyter notebook cell and run it.
#
# Setup (once):
#   pip install ollama
#   ollama pull llama3.2


# %% CELL 0: Shared setup (run this first)

from ollama import chat

MODEL = "llama3.2"

# One customer review used in every technique below
REVIEW = "The pizza was cold and the waiter ignored us for 20 minutes."


def ask(prompt):
    """Send a prompt to the model and print the reply."""
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.message.content)
    print()


# %% CELL 1: Vague prompt vs Clear prompt
# Tip: Say WHAT you want, HOW long, and WHAT style.

print("=== Vague prompt (weak) ===")
ask("Reply to this review: " + REVIEW)

print("=== Clear prompt (better) ===")
ask(
    f"""Write a short reply to this cafe review.

Rules:
- Be polite and apologetic
- Offer a free dessert next visit
- Maximum 2 sentences

Review: {REVIEW}"""
)


# %% CELL 2: Role prompting
# Tip: Tell the model WHO it should act as.

ask(
    f"""You are the friendly manager of Campus Cafe.
Write a short reply to this customer review.
Be warm, take responsibility, and invite them back.

Review: {REVIEW}"""
)


# %% CELL 3: Few-shot prompting
# Tip: Show a few examples, then ask for a new one.

ask(
    f"""Write a short polite reply to each cafe review.

Example 1:
Review: The coffee was amazing!
Reply: Thank you! We are glad you enjoyed our coffee. See you again soon!

Example 2:
Review: The sandwich was too salty.
Reply: Sorry about that. Please visit again and we will make it right for you.

Now write a reply for this review:
Review: {REVIEW}
Reply:"""
)


# %% CELL 4: Step-by-step prompting (Chain of Thought)
# Tip: Ask the model to think in steps before the final answer.

ask(
    f"""A customer left this review for Campus Cafe:
"{REVIEW}"

Think step by step:
1. What is the main problem?
2. How should the cafe respond?
3. Write the final short reply (2 sentences)."""
)


# %% CELL 5: Output format control
# Tip: Tell the model the exact format you want.

ask(
    f"""Read this cafe review and reply in EXACTLY this format:

Sentiment: Positive / Negative / Neutral
Issue: <main problem in a few words>
Reply: <one short polite sentence>

Review: {REVIEW}"""
)
