import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def normalize_text(text):
    if not isinstance(text, str): return "Unknown"
    text = text.lower().replace('/', ' ').replace('-', ' ').strip()
    if "conceptual" in text: return "Conceptual Error"
    elif "algebraic" in text: return "Algebraic Error"
    elif "arithmetic" in text: return "Arithmetic Error"
    elif "format" in text or "extract" in text: return "Formatting Error"
    return "Other"

def main():
    print("Loading data...")
    human_data = load_json('manual_analysis.json')
    gemini_data = load_json('gemini_error_taxonomy.json')
    groq_data = load_json('groq_error_taxonomy.json')

    df_human = pd.DataFrame(human_data).rename(columns={'error_type': 'Human'})
    df_gemini = pd.DataFrame(gemini_data).rename(columns={'error_type': 'Gemini'})
    df_groq = pd.DataFrame(groq_data).rename(columns={'error_type': 'Groq (Llama-3.1)'})

    for df, col in [(df_human, 'Human'), (df_gemini, 'Gemini'), (df_groq, 'Groq (Llama-3.1)')]:
        df[col] = df[col].apply(normalize_text)

    # Force data types to match perfectly for merging
    df_human['id'] = df_human['id'].astype(int)
    df_gemini['id'] = df_gemini['id'].astype(int)
    df_groq['id'] = df_groq['id'].astype(int)
    
    df_human['model'] = df_human['model'].astype(str).str.strip().str.upper()
    df_gemini['model'] = df_gemini['model'].astype(str).str.strip().str.upper()
    df_groq['model'] = df_groq['model'].astype(str).str.strip().str.upper()

    merged_df = pd.merge(df_human[['id', 'model', 'Human']], df_gemini[['id', 'model', 'Gemini']], on=['id', 'model'], how='inner')
    merged_df = pd.merge(merged_df, df_groq[['id', 'model', 'Groq (Llama-3.1)']], on=['id', 'model'], how='inner').dropna()

    print(f"Data merged successfully! Found {len(merged_df)} overlapping evaluations.")

    # ==========================================
    # PLOT 1: Accuracy Bar Chart
    # ==========================================
    gemini_acc = (merged_df['Gemini'] == merged_df['Human']).mean() * 100
    groq_acc = (merged_df['Groq (Llama-3.1)'] == merged_df['Human']).mean() * 100

    plt.figure(figsize=(6, 5))
    models = ['Gemini 3.1 Pro', 'Llama 3.1 8B (Groq)']
    accuracies = [gemini_acc, groq_acc]
    
    # Fix the Seaborn warning using hue and legend=False
    ax = sns.barplot(
        x=models, 
        y=accuracies, 
        hue=models,
        palette=['#4285F4', '#FF8C00'],
        legend=False
    )
    
    plt.title('Evaluator Accuracy vs. Human Ground Truth', fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add percentage text on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), textcoords='offset points')
    
    plt.tight_layout()
    # Save as PNG instead of PDF!
    plt.savefig('evaluator_accuracy.png', format='png', dpi=300)
    print("Saved evaluator_accuracy.png")
    plt.close() # Flush the plot from memory

    # ==========================================
    # PLOT 2 & 3: Confusion Matrices
    # ==========================================
    categories = ["Conceptual Error", "Algebraic Error", "Arithmetic Error", "Formatting Error"]
    
    def plot_confusion(evaluator_col, filename, title):
        plt.figure(figsize=(7, 6))
        cm = pd.crosstab(merged_df['Human'], merged_df[evaluator_col])
        cm = cm.reindex(index=categories, columns=categories, fill_value=0)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    linewidths=1, linecolor='black')
        plt.title(title, fontweight='bold')
        plt.ylabel('True Error (Human)')
        plt.xlabel(f'Predicted Error ({evaluator_col})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=300)
        print(f"Saved {filename}")
        plt.close() # Flush the plot from memory

    plot_confusion('Gemini', 'gemini_confusion.png', 'Confusion Matrix: Human vs Gemini')
    plot_confusion('Groq (Llama-3.1)', 'groq_confusion.png', 'Confusion Matrix: Human vs Llama 3.1')

if __name__ == "__main__":
    main()