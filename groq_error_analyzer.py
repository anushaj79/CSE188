import json
import time
import os
from prompt_utils import prompt_groq

# --- GROQ "TEACHER" PROMPT ---
# Notice how we heavily restrict the output to save tokens
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

def parse_groq_analysis(text):
    """Extract the structured data directly from Groq's minimal JSON response."""
    try:
        # Strip out markdown code blocks just in case the model disobeys
        clean_json_str = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json_str)
        return {
            "error_type": data.get("error_type", "Uncategorized"),
            "failed_step": data.get("failed_step", "Unknown")
        }
    except json.JSONDecodeError:
        print(f"  ⚠️ Failed to parse JSON from model output: {text}")
        return {
            "error_type": "Parsing Error",
            "failed_step": "Unknown"
        }

def run_groq_analysis():
    with open("model_answers.json", "r") as f:
        data = json.load(f)

    results_file = "groq_error_taxonomy.json"
    tracker_file = "groq_processed_ids.json"
    
    # Load existing results (the flat list)
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            flat_results = json.load(f)
    else:
        flat_results = []

    # Load tracker so we don't re-process questions where models got it right
    if os.path.exists(tracker_file):
        with open(tracker_file, "r") as f:
            processed_ids = set(json.load(f))
    else:
        processed_ids = set()

    print(f"Loaded {len(data)} questions. Found {len(processed_ids)} already processed.")
    print("Starting automated error analysis... (Optimized for minimal output tokens)")

    for idx, item in enumerate(data):
        q_id = int(item.get("question_id", idx))
        
        # Skip if we already finished this one
        if q_id in processed_ids:
            continue
            
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        for model in ["gpt", "claude"]:
            if model not in item or not item[model].get("final_answer"):
                continue
                
            prediction = item[model]["final_answer"]
            steps = item[model]["steps"]
            
            # If the model got it WRONG, ask Groq to analyze it
            if not is_correct(prediction, ground_truth):
                print(f"Asking Llama-3 to analyze {model.upper()} failure on Q{q_id}...")
                
                eval_prompt = f"Problem: {question}\nGround Truth: {ground_truth}\nStudent's Wrong Answer: {prediction}\n\nStudent Steps:\n"
                for i, step in enumerate(steps, 1):
                    eval_prompt += f"Step {i}: {step}\n"
                
                # --- ROBUST RETRY LOOP FOR RATE LIMITS ---
                success = False
                while not success:
                    try:
                        analysis_text = prompt_groq(eval_prompt, TAXONOMY_INSTRUCTION)
                        parsed = parse_groq_analysis(analysis_text)
                        
                        # Python handles injecting the ID and Model into the final flat list!
                        flat_results.append({
                            "id": q_id,
                            "model": model.upper(),
                            "error_type": parsed["error_type"],
                            "failed_step": parsed["failed_step"]
                        })
                        
                        success = True
                        
                        # Add a small buffer between requests to avoid hitting the Tokens Per Minute (TPM) limit
                        time.sleep(2) 
                        
                    except Exception as e:
                        error_msg = str(e)
                        # Catch the 429 Too Many Requests error
                        if "429" in error_msg or "rate limit" in error_msg.lower():
                            print("  ⚠️ Free tier rate limit (TPM/RPM) hit! Pausing for 60 seconds...")
                            time.sleep(60)
                        else:
                            print(f"  ❌ Unknown API Error: {error_msg}")
                            flat_results.append({
                                "id": q_id,
                                "model": model.upper(),
                                "error_type": "API Error",
                                "failed_step": "Unknown"
                            })
                            break # Break out of while loop on non-quota errors

        # Mark this question as completely processed
        processed_ids.add(q_id)
        
        # Save after every single question so you never lose progress
        with open(results_file, "w") as f:
            json.dump(flat_results, f, indent=2)
            
        with open(tracker_file, "w") as f:
            json.dump(list(processed_ids), f)

    print("\n✅ Groq Analysis complete! Check groq_error_taxonomy.json for the formatted results.")

if __name__ == "__main__":
    run_groq_analysis()