import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set plotting style for professional academic papers
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def extract_step_number(step_string):
    """Extracts the integer from strings like 'Step 9'."""
    if not isinstance(step_string, str):
        return None
    match = re.search(r'\d+', step_string)
    return int(match.group()) if match else None

def load_data():
    # Load Groq (Llama-3) data
    with open("groq_error_taxonomy.json", "r") as f:
        groq_data = json.load(f)
    df_groq = pd.DataFrame(groq_data)
    df_groq.rename(columns={'error_type': 'groq_error', 'failed_step': 'groq_step'}, inplace=True)

    # Load Gemini data
    with open("gemini_error_taxonomy.json", "r") as f: # Ensure your file is named this
        gemini_data = json.load(f)
    df_gemini = pd.DataFrame(gemini_data)
    df_gemini.rename(columns={'error_type': 'gemini_error', 'failed_step': 'gemini_step'}, inplace=True)

    # Merge datasets on Question ID and Model
    df_merged = pd.merge(df_groq, df_gemini, on=['id', 'model'], how='inner')
    
    # Extract numerical steps
    df_merged['groq_step_num'] = df_merged['groq_step'].apply(extract_step_number)
    df_merged['gemini_step_num'] = df_merged['gemini_step'].apply(extract_step_number)
    
    return df_merged

def calculate_agreement(df):
    """Calculate how often Gemini and Llama-3 agreed."""
    print("--- INTER-ANNOTATOR AGREEMENT ---")
    
    # Error Type Agreement
    type_agreement = (df['groq_error'] == df['gemini_error']).mean() * 100
    print(f"Error Type Agreement: {type_agreement:.2f}%")
    
    # Step Number Agreement (Exact Match)
    step_agreement = (df['groq_step_num'] == df['gemini_step_num']).mean() * 100
    print(f"Exact Step Match Agreement: {step_agreement:.2f}%")
    
    # Step Number Agreement (Within 1 step)
    df['step_diff'] = abs(df['groq_step_num'] - df['gemini_step_num'])
    step_close_agreement = (df['step_diff'] <= 1).mean() * 100
    print(f"Step Match (±1 step) Agreement: {step_close_agreement:.2f}%\n")

def plot_error_distribution(df):
    """Generates a stacked bar chart of error types by model."""
    print("Generating Error Distribution Chart...")
    
    # We will use Groq's classification for this chart as the primary analysis
    # (You can swap 'groq_error' to 'gemini_error' if you prefer)
    error_counts = df.groupby(['model', 'groq_error']).size().unstack(fill_value=0)
    
    # Plot
    ax = error_counts.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')
    plt.title("Distribution of Mathematical Reasoning Errors by Model", pad=15)
    plt.xlabel("Language Model", labelpad=10)
    plt.ylabel("Number of Errors", labelpad=10)
    plt.xticks(rotation=0)
    plt.legend(title="Error Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save for LaTeX
    plt.savefig("fig1_error_distribution.png", dpi=300)
    plt.close()

def plot_logic_decay(df):
    """Generates a KDE/Histogram showing exactly where logic breaks down."""
    print("Generating Logic Decay Chart...")
    
    plt.figure(figsize=(8, 6))
    
    # Plot distribution of the step numbers where failure occurred
    sns.histplot(data=df, x='groq_step_num', hue='model', multiple='dodge', 
                 bins=range(1, int(df['groq_step_num'].max()) + 2), 
                 palette='Set2', shrink=0.8)
    
    plt.title("Logic Decay: Step Number Where Initial Failure Occurs", pad=15)
    plt.xlabel("Step Number", labelpad=10)
    plt.ylabel("Frequency", labelpad=10)
    plt.xlim(0, max(10, df['groq_step_num'].max() + 1))
    plt.tight_layout()
    
    # Save for LaTeX
    plt.savefig("fig2_logic_decay.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    df = load_data()
    
    # 1. Print Paper Statistics
    print(f"Total failure instances analyzed: {len(df)}")
    print(f"GPT Failures: {len(df[df['model'] == 'GPT'])}")
    print(f"Claude Failures: {len(df[df['model'] == 'CLAUDE'])}\n")
    
    calculate_agreement(df)
    
    # 2. Generate PDF/PNG figures for LaTeX
    plot_error_distribution(df)
    plot_logic_decay(df)
    
    print("✅ All analysis complete! Images saved as 'fig1_error_distribution.png' and 'fig2_logic_decay.png'.")