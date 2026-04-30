import json
import time
import os
from prompt_utils import load_dataset, prompt_gpt, prompt_claude, extract_answer_tag, split_steps

DATASET_URL = "https://raw.githubusercontent.com/sarahmart/HARDMath/refs/heads/main/evaluation/data/HARDMath_mini.json"
NUM_QUESTIONS = 365
INSTRUCTION = (
    "You are a math expert. Solve the following problem step-by-step. "
    "Make sure to include all calculations for each step. Do not apologize or refuse.\n"
    "Format your response EXACTLY as shown below. Do not include any text outside this format.\n\n"
    "[Steps]\n"
    "<step>Step 1: Explanation and calculation.</step>\n"
    "<step>Step 2: Explanation and calculation.</step>\n"
    "...\n"
    "<step>Final Step: Explanation and calculation.</step>\n\n"
    "[Answer]\n"
    "<answer>Final Answer in required format</answer>\n"
)

SAVE_FILE = "model_answers.json"

def load_existing_answers():
    """Load existing answers if the file exists, otherwise return {}."""
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            try:
                data = json.load(f)
                # Map by question text for quick lookup
                return {item["question"]: item for item in data}
            except json.JSONDecodeError:
                return {}
    return {}

def main():
    dataset = load_dataset(DATASET_URL)
    questions = dataset[:NUM_QUESTIONS]  # strict order
    existing_answers = load_existing_answers()

    print(f"🔄 Resuming... found {len(existing_answers)} previously completed questions")

    total_start = time.time()

    for idx, q in enumerate(questions, start=0): 
        question_text = q.get("question", "")
        ans_record = existing_answers.get(question_text, {
            "question_id": str(idx),
            "question": question_text,
            "ground_truth": q.get("answer_val", "").strip(),
            "question_type": q.get("question_type", "").strip(),
            "answer_type": q.get("answer_type", "").strip()
        })

        print(f"\n📝 Question {idx}/{len(questions)}: {question_text[:80]}{'...' if len(question_text) > 80 else ''}")

        question_start = time.time()

        for model_name, prompt_func in [("gpt", prompt_gpt), ("claude", prompt_claude)]:
            # Skip if model already has a final answer
            if model_name in ans_record and ans_record[model_name].get("final_answer") is not None:
                print(f"⏭️ Skipping {model_name} for Question {idx} (already completed)")
                continue

            try:
                model_start = time.time()
                output = prompt_func(question_text, INSTRUCTION)
                model_time = time.time() - model_start

                final_answer = extract_answer_tag(output)
                steps_list = split_steps(output)

                ans_record[model_name] = {
                    "steps": steps_list,
                    "final_answer": final_answer,
                }
                print(f"✅ Finished {model_name} in {model_time:.2f} sec")
            except Exception as e:
                print(f"⚠️ {model_name} error on Question {idx}: {e}")
                ans_record[model_name] = {"steps": [], "final_answer": None}

        question_time = time.time() - question_start
        print(f"⏱️ Total time for Question {idx}: {question_time:.2f} sec")

        # Save progress after each question
        existing_answers[question_text] = ans_record
        with open(SAVE_FILE, "w") as f:
            json.dump(list(existing_answers.values()), f, indent=2, ensure_ascii=False)

    total_time = time.time() - total_start
    print(f"\n🎉 All {len(existing_answers)} questions processed in {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()
