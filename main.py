import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "prithivida/grammar_error_correcter_v1"

# Load the tokenizer and model once at the start
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def correct_and_style_text(text: str, style: str = "general") -> str:
    """
    Takes input text, corrects grammar, and attempts to rewrite in the specified style
    ('business', 'general', or 'casual') by using prompt engineering.
    """
    # Prompt: The 'gec:' prefix is often used with this model to indicate grammar correction.
    # We'll also include a style instruction. 
    # Feel free to tweak the wording if you want more or fewer changes in tone.
    prompt = f"gec: Rewrite the following text in a {style} style, ensuring correct grammar and provide better vocabulary: {text}"

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    # Decode
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text


def main():
    print("============================================================")
    print(" Welcome to the Grammar & Style Improver (T5-based)         ")
    print(" Paste (or type) your text below, then choose a style.      ")
    print(" Type 'exit' or 'quit' to stop.                             ")
    print("============================================================\n")

    while True:
        # 1) Get user input text
        user_text = input("Enter text to correct/improve (or 'exit' to quit): ")
        if user_text.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not user_text.strip():
            print("Please enter some text.")
            continue

        # 2) Prompt for style
        style = input("Choose style: 'business', 'general', or 'casual': ").lower().strip()
        if style not in ["business", "general", "casual"]:
            print("Unrecognized style. Defaulting to 'general' style.")
            style = "general"

        # 3) Correct and rewrite
        corrected_output = correct_and_style_text(user_text, style)

        # 4) Print result
        print("\n--- Corrected/Rewritten Text ---")
        print(corrected_output)
        print("--------------------------------\n")


if __name__ == "__main__":
    main()
