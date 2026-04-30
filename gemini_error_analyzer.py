import json
import time
import re
import os
from prompt_utils import prompt_gemini

# --- GEMINI "TEACHER" PROMPTS ---
TAXONOMY_INSTRUCTION = (
    "You are an expert mathematics professor. You are grading a student's step-by-step work. "
    "The student reached the WRONG final answer. "
    "Analyze their steps, compare it to the ground truth, and identify the EXACT step where the FIRST error occurred.\n\n"
    "You MUST categorize the error into EXACTLY ONE of the following categories:\n"
    "1. Arithmetic Error\n"
    "2. Algebraic Error\n"
    "3. Conceptual Setup Error\n"
    "4. Formatting/Extraction Error\n\n"
    "Return ONLY a raw JSON object. Do not include markdown formatting, explanations, or any other text. "
    "Use this exact format:\n"
    '{"error_type": "Category Name", "failed_step": "Step X"}'
)

def clean_string(s):
    if not s: return ""
    return s.replace(" ", "").replace("\\(", "").replace("\\)", "").strip()

def is_correct(prediction, ground_truth):
    return clean_string(prediction) == clean_string(ground_truth)

def parse_gemini_analysis(text):
    """Extract the structured data from Gemini's response."""
    step_match = re.search(r"\[Error Step\]\s*(?:Step\s*)?(\d+)", text, re.IGNORECASE)
    cat_match = re.search(r"\[Category\]\s*(.*)", text, re.IGNORECASE)
    
    return {
        "error_step": int(step_match.group(1)) if step_match else None,
        "category": cat_match.group(1).strip() if cat_match else "Uncategorized",
        "raw_analysis": text
    }

def run_gemini_analysis():
    with open("model_answers.json", "r") as f:
        data = json.load(f)

    results_file = "gemini_error_taxonomy.json"
    
    # Load existing progress so we don't waste our 5 RPM quota on finished questions
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            analyzed_data = json.load(f)
    else:
        analyzed_data = []

    # Track which questions we've already done
    processed_ids = {item["question_id"] for item in analyzed_data}

    print(f"Loaded {len(data)} questions. Found {len(processed_ids)} already processed.")
    print("Starting automated error analysis (max 5 Requests Per Minute)...")

    for idx, item in enumerate(data):
        q_id = item["question_id"]
        
        # Skip if we already finished this one
        if q_id in processed_ids:
            continue
            
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        item_analysis = {"question_id": q_id, "models": {}}
        
        for model in ["gpt", "claude"]:
            if model not in item or not item[model].get("final_answer"):
                continue
                
            prediction = item[model]["final_answer"]
            steps = item[model]["steps"]
            
            # If the model got it WRONG, ask Gemini to analyze it
            if not is_correct(prediction, ground_truth):
                print(f"Analyzing {model} failure on Q{idx}...")
                
                eval_prompt = f"Problem: {question}\nGround Truth: {ground_truth}\nStudent's Wrong Answer: {prediction}\n\nStudent Steps:\n"
                for i, step in enumerate(steps, 1):
                    eval_prompt += f"Step {i}: {step}\n"
                
                # --- ROBUST RETRY LOOP ---
                success = False
                while not success:
                    try:
                        analysis_text = prompt_gemini(eval_prompt, TAXONOMY_INSTRUCTION)
                        item_analysis["models"][model] = parse_gemini_analysis(analysis_text)
                        success = True
                        
                        # Wait 15 seconds to strictly enforce the 5 RPM limit
                        print("  Success! Sleeping 15s to respect quota...")
                        time.sleep(15) 
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "Quota" in error_msg:
                            print("  ⚠️ Rate limit hit. Waiting 60 seconds before retrying...")
                            time.sleep(60)
                        else:
                            print(f"  ❌ Unknown API Error: {error_msg}")
                            item_analysis["models"][model] = {"status": "Error", "details": error_msg}
                            break # Break out of while loop on non-quota errors
            else:
                item_analysis["models"][model] = {"status": "Correct"}
                
        analyzed_data.append(item_analysis)
        
        # Save after every single question so you never lose progress
        with open(results_file, "w") as f:
            json.dump(analyzed_data, f, indent=2)

    print("\n✅ Analysis complete! Check gemini_error_.json")

if __name__ == "__main__":
    run_gemini_analysis()