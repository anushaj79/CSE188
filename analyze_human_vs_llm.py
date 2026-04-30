import json
import pandas as pd

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def normalize_text(text):
    """Standardizes error category strings so they match across JSONs."""
    if not isinstance(text, str):
        return "Unknown"
    text = text.lower().replace('/', ' ').replace('-', ' ').strip()
    if "conceptual" in text:
        return "conceptual setup error"
    elif "algebraic" in text:
        return "algebraic error"
    elif "arithmetic" in text:
        return "arithmetic error"
    elif "format" in text or "extract" in text:
        return "formatting extraction error"
    return text

def main():
    # 1. Load the data
    human_data = load_json('manual_analysis.json')
    gemini_data = load_json('gemini_error_taxonomy.json')
    groq_data = load_json('groq_error_taxonomy.json')

    # 2. Convert to DataFrames
    df_human = pd.DataFrame(human_data)
    df_gemini = pd.DataFrame(gemini_data)
    df_groq = pd.DataFrame(groq_data)

    # 3. Rename and Normalize Columns
    df_human = df_human.rename(columns={'error_type': 'human_error', 'failed_step': 'human_step'})
    df_gemini = df_gemini.rename(columns={'error_type': 'gemini_error', 'failed_step': 'gemini_step'})
    df_groq = df_groq.rename(columns={'error_type': 'groq_error', 'failed_step': 'groq_step'})

    for df, col in [(df_human, 'human_error'), (df_gemini, 'gemini_error'), (df_groq, 'groq_error')]:
        df[col] = df[col].apply(normalize_text)

    # 4. Merge DataFrames ONLY on the items the Human graded
    merged_df = pd.merge(df_human, df_gemini[['id', 'model', 'gemini_error']], on=['id', 'model'], how='left')
    merged_df = pd.merge(merged_df, df_groq[['id', 'model', 'groq_error']], on=['id', 'model'], how='left')

    # Drop any rows where an LLM failed to return a valid JSON response
    merged_df = merged_df.dropna()

    # 5. Calculate Accuracies (LLM vs Human Ground Truth)
    gemini_accuracy = (merged_df['gemini_error'] == merged_df['human_error']).mean() * 100
    groq_accuracy = (merged_df['groq_error'] == merged_df['human_error']).mean() * 100
    llm_agreement = (merged_df['gemini_error'] == merged_df['groq_error']).mean() * 100

    print("==================================================")
    print("🏆 EVALUATOR RELIABILITY (VS HUMAN GROUND TRUTH)")
    print("==================================================")
    print(f"Total human-graded samples evaluated: {len(merged_df)}")
    print(f"Gemini 1.5 Flash Accuracy: {gemini_accuracy:.1f}%")
    print(f"Groq (Llama 3) Accuracy:   {groq_accuracy:.1f}%")
    print(f"LLM-to-LLM Agreement:      {llm_agreement:.1f}%")

    # 6. Generate Confusion Matrices for the Paper
    print("\n==================================================")
    print("📊 CONFUSION MATRIX: HUMAN vs GEMINI")
    print("Rows = True Human Label, Columns = Gemini Prediction")
    print("==================================================")
    print(pd.crosstab(merged_df['human_error'], merged_df['gemini_error']))

    print("\n==================================================")
    print("📊 CONFUSION MATRIX: HUMAN vs GROQ")
    print("Rows = True Human Label, Columns = Groq Prediction")
    print("==================================================")
    print(pd.crosstab(merged_df['human_error'], merged_df['groq_error']))

    # 7. Find Qualitative Examples for the Paper
    print("\n==================================================")
    print("🔎 QUALITATIVE EXAMPLES TO PUT IN YOUR PAPER")
    print("Cases where Gemini, Groq, and Human ALL disagreed:")
    print("==================================================")
    total_disagreement = merged_df[
        (merged_df['human_error'] != merged_df['gemini_error']) & 
        (merged_df['human_error'] != merged_df['groq_error']) & 
        (merged_df['gemini_error'] != merged_df['groq_error'])
    ]
    
    if not total_disagreement.empty:
        print(total_disagreement[['id', 'model', 'human_error', 'gemini_error', 'groq_error']])
    else:
        print("No cases found where all 3 completely disagreed. Looking for cases where BOTH LLMs were wrong...")
        both_wrong = merged_df[
            (merged_df['human_error'] != merged_df['gemini_error']) & 
            (merged_df['human_error'] != merged_df['groq_error'])
        ]
        print(both_wrong[['id', 'model', 'human_error', 'gemini_error', 'groq_error']].head(5))

if __name__ == "__main__":
    main()