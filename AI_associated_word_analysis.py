from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
EXCEL_PATH = PROJECT_ROOT / "data" / "JAPPL_NIH_2020_2026.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "results"

PROJECTS = ["JAPPL", "NIH"]
YEARS = ["2020", "2026"]

PROJECT_METADATA_COLUMNS = {
    "JAPPL": ["PMID", "DOI"],
    "NIH": ["Project Number", "Grant_type"],
}

PROJECT_HEATMAP_TITLES = {
    "JAPPL": "JAPPL publications",
    "NIH": "NIH grant abstracts",
}

WORD_LIST_SHEET = "AI_associated_words"
WORD_LIST_COLUMN = "Word"
TOP_N_WORDS = 20

TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def load_AI_associated_words(
    excel_path: Path,
    sheet_name: str,
    column_name: str
) -> list[str]:
    df_words = pd.read_excel(excel_path, sheet_name=sheet_name)

    if column_name not in df_words.columns:
        raise ValueError(
            f"Column '{column_name}' not found in sheet '{sheet_name}'. "
            f"Available columns: {list(df_words.columns)}"
        )

    words = (
        df_words[column_name]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .tolist()
    )

    if not words:
        raise ValueError(
            f"No valid AI-associated words were found in "
            f"'{sheet_name}' column '{column_name}'."
        )

    return words

AI_ASSOCIATED_WORDS = load_AI_associated_words(
        EXCEL_PATH,
        sheet_name=WORD_LIST_SHEET,
        column_name=WORD_LIST_COLUMN
    )

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def combine_manuscript_text(df: pd.DataFrame) -> pd.Series:
    cols = ["Manuscript_text_1", "Manuscript_text_2", "Manuscript_text_3"]
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    return (
        df[cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(normalize_spaces)
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def build_word_patterns(words: list[str]) -> dict[str, re.Pattern]:
    patterns = {}
    for word in words:
        variants = [w.strip() for w in word.split("/") if w.strip()]
        escaped_variants = [re.escape(v) for v in variants]
        pattern = re.compile(
            rf"\b(?:{'|'.join(escaped_variants)})\b",
            flags=re.IGNORECASE
        )
        patterns[word] = pattern
    return patterns


WORD_PATTERNS = build_word_patterns(AI_ASSOCIATED_WORDS)


def count_AI_associated_words(
    text: str,
    patterns: dict[str, re.Pattern]
) -> dict[str, int]:
    return {word: len(pattern.findall(text)) for word, pattern in patterns.items()}


def safe_divide(numerator: float, denominator: float, multiplier: float = 1.0) -> float:
    if denominator == 0:
        return np.nan
    return (numerator / denominator) * multiplier


def relative_change_percent(old: float, new: float) -> float:
    if old == 0:
        if new == 0:
            return np.nan
        return np.inf
    return ((new - old) / old) * 100


def get_sheet_name(project: str, year: str) -> str:
    return f"{project}_{year}"


def create_density_heatmap(
    word_summary_df: pd.DataFrame,
    project: str,
    output_dir: Path,
    top_n: int = 20
) -> Path:
    """
    Creates a heatmap using Density_2020_per_10000 and Density_2026_per_10000
    for the top N words ranked by % relative change.

    Rows = AI-associated words
    Columns = 2020 and 2026 density per 10,000 words
    """

    required_cols = [
        "Word",
        "Density_2020_per_10000",
        "Density_2026_per_10000",
        "% relative change",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in word_summary_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot create heatmap for {project}. "
            f"Missing columns: {missing_cols}"
        )

    plot_df = word_summary_df.copy()

    # Replace infinite relative changes with NaN only for ranking stability.
    # This avoids problems when a word has density = 0 in 2020 but >0 in 2026.
    plot_df["_relative_change_for_sorting"] = (
        plot_df["% relative change"]
        .replace([np.inf, -np.inf], np.nan)
    )

    # Prioritize words with large relative changes, but keep words with infinite
    # change by sorting separately if needed.
    infinite_change_df = plot_df[
        plot_df["% relative change"].isin([np.inf, -np.inf])
    ].copy()

    finite_change_df = plot_df[
        ~plot_df["% relative change"].isin([np.inf, -np.inf])
    ].copy()

    finite_change_df = finite_change_df.sort_values(
        by="_relative_change_for_sorting",
        ascending=False,
        na_position="last"
    )

    plot_df = pd.concat(
        [infinite_change_df, finite_change_df],
        axis=0
    ).head(top_n)

    heatmap_df = (
        plot_df[
            [
                "Word",
                "Density_2020_per_10000",
                "Density_2026_per_10000",
            ]
        ]
        .set_index("Word")
        .rename(
            columns={
                "Density_2020_per_10000": "2020",
                "Density_2026_per_10000": "2026",
            }
        )
    )

    fig_height = max(4, 0.35 * len(heatmap_df))
    fig_width = 6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        heatmap_df.values,
        aspect="auto"
    )

    ax.set_xticks(np.arange(heatmap_df.shape[1]))
    ax.set_xticklabels(heatmap_df.columns)

    ax.set_yticks(np.arange(heatmap_df.shape[0]))
    ax.set_yticklabels(heatmap_df.index)

    project_heatmap_title = PROJECT_HEATMAP_TITLES.get(project)

    ax.set_title(
        f"{project_heatmap_title} - top {len(heatmap_df)} words\n"
        f"Density per 10,000 words"
    )

    cbar = fig.colorbar(im, ax=ax)

    fig.tight_layout()

    output_path = output_dir / f"{project}_density_heatmap_top_{top_n}_words.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def summarize_top_words(
    project_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    top_n: int = 20
) -> pd.DataFrame:
    summary_rows = []

    for project, (_, word_summary_df) in project_outputs.items():
        top_words = (
            word_summary_df
            .dropna(subset=["% relative change"])
            .nlargest(top_n, "% relative change")
            .copy()
        )

        summary_rows.append({
            "Project": project,
            "Top_n_words": len(top_words),
            "Mean_%_relative_change_top_words": top_words["% relative change"].replace([np.inf, -np.inf], np.nan).mean(),
            "Median_%_relative_change_top_words": top_words["% relative change"].replace([np.inf, -np.inf], np.nan).median(),
            "Min_%_relative_change_top_words": top_words["% relative change"].replace([np.inf, -np.inf], np.nan).min(),
            "Max_%_relative_change_top_words": top_words["% relative change"].replace([np.inf, -np.inf], np.nan).max(),
        })

    return pd.DataFrame(summary_rows)


def analyze_project(
    project: str,
    workbook_data: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manuscript_rows = []

    for year_name in YEARS:
        sheet_name = get_sheet_name(project, year_name)
        if sheet_name not in workbook_data:
            raise ValueError(f"Sheet '{sheet_name}' not found in workbook.")

        df = workbook_data[sheet_name].copy()

        for col in ["Title", "Abstract"]:
            if col not in df.columns:
                df[col] = ""

        df["Manuscript_text"] = combine_manuscript_text(df)

        for idx, row in df.iterrows():
            title_text = clean_text(row.get("Title", ""))
            abstract_text = clean_text(row.get("Abstract", ""))
            manuscript_text = clean_text(row.get("Manuscript_text", ""))

            combined_text = normalize_spaces(" ".join([title_text, abstract_text, manuscript_text]))

            title_wc = word_count(title_text)
            abstract_wc = word_count(abstract_text)
            manuscript_wc = word_count(manuscript_text)
            total_wc = word_count(combined_text)

            title_counts = count_AI_associated_words(title_text, WORD_PATTERNS)
            abstract_counts = count_AI_associated_words(abstract_text, WORD_PATTERNS)
            manuscript_counts = count_AI_associated_words(manuscript_text, WORD_PATTERNS)

            combined_counts = {
                word: title_counts[word] + abstract_counts[word] + manuscript_counts[word]
                for word in AI_ASSOCIATED_WORDS
            }

            title_occ = sum(title_counts.values())
            abstract_occ = sum(abstract_counts.values())
            manuscript_occ = sum(manuscript_counts.values())
            total_occ = sum(combined_counts.values())
            unique_AI_associated_words_used = sum(v > 0 for v in combined_counts.values())
            AI_associated_word_diversity_ratio = safe_divide(
                unique_AI_associated_words_used,
                len(AI_ASSOCIATED_WORDS)
            )
            occurrences_per_10000 = safe_divide(total_occ, total_wc, 10000)
            
            metadata_cols = PROJECT_METADATA_COLUMNS.get(project, [])

            out = {
                "Project": project,
                "Sheet_Year": str(year_name),
                "Source_sheet": sheet_name,
                "Row_in_sheet": idx + 2,
                "No": row.get("No", np.nan),
            }
            
            for col in metadata_cols:
                out[col] = row.get(col, "")
            
            out.update({
                "Title_word_count": title_wc,
                "Abstract_word_count": abstract_wc,
                "Manuscript_word_count": manuscript_wc,
                "Combined_word_count_Title_Abstract_Manuscript": total_wc,
                "Title_AI-associated_occurrences": title_occ,
                "Abstract_AI-associated_occurrences": abstract_occ,
                "Manuscript_AI-associated_occurrences": manuscript_occ,
                "Combined_AI-associated_occurrences": total_occ,
                "Combined_occurrences_per_10000_words": occurrences_per_10000,
                "Unique_AI-associated_words_used": unique_AI_associated_words_used,
                "AI-associated_word_diversity_ratio": AI_associated_word_diversity_ratio,
                "Any_AI-associated_word_present": int(total_occ > 0),
            })
            
            for word in AI_ASSOCIATED_WORDS:
                out[f"count__{word}"] = combined_counts[word]

            manuscript_rows.append(out)

    manuscripts_df = pd.DataFrame(manuscript_rows)

    total_word_count_2020 = manuscripts_df.loc[
        manuscripts_df["Sheet_Year"] == "2020",
        "Combined_word_count_Title_Abstract_Manuscript"
    ].sum()

    total_word_count_2026 = manuscripts_df.loc[
        manuscripts_df["Sheet_Year"] == "2026",
        "Combined_word_count_Title_Abstract_Manuscript"
    ].sum()

    sub_2020 = manuscripts_df[manuscripts_df["Sheet_Year"] == "2020"]
    sub_2026 = manuscripts_df[manuscripts_df["Sheet_Year"] == "2026"]

    word_rows = []
    for word in AI_ASSOCIATED_WORDS:
        count_col = f"count__{word}"

        occ_2020 = sub_2020[count_col].sum()
        occ_2026 = sub_2026[count_col].sum()

        density_2020 = safe_divide(occ_2020, total_word_count_2020, 10000)
        density_2026 = safe_divide(occ_2026, total_word_count_2026, 10000)

        word_rows.append({
            "Project": project,
            "Word": word,
            "2020_total_occurrences": occ_2020,
            "2026_total_occurrences": occ_2026,
            "Total_word_count_2020": total_word_count_2020,
            "Total_word_count_2026": total_word_count_2026,
            "Density_2020_per_10000": density_2020,
            "Density_2026_per_10000": density_2026,
            "% relative change": relative_change_percent(density_2020, density_2026),
        })

    word_summary_df = pd.DataFrame(word_rows).sort_values(
        by="% relative change",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    return manuscripts_df, word_summary_df


def main() -> None:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {EXCEL_PATH}\n"
            "Place the Excel file in the 'data' folder before running the script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_sheets = [get_sheet_name(project, year) for project in PROJECTS for year in YEARS]
    workbook_data = pd.read_excel(EXCEL_PATH, sheet_name=required_sheets)

    project_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    heatmap_paths = []

    for project in PROJECTS:
        manuscripts_df, word_summary_df = analyze_project(project, workbook_data)
        project_outputs[project] = (manuscripts_df, word_summary_df)
        heatmap_path = create_density_heatmap(word_summary_df=word_summary_df,
            project=project, output_dir=OUTPUT_DIR, top_n=TOP_N_WORDS)

        heatmap_paths.append(heatmap_path)

    top_word_summary_df = summarize_top_words(project_outputs, top_n=TOP_N_WORDS)

    output_excel_path = OUTPUT_DIR / "AI_associated_word_analysis_results.xlsx"

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        for project in PROJECTS:
            manuscripts_df, word_summary_df = project_outputs[project]
            manuscripts_df.to_excel(writer, sheet_name=f"{project}_manuscript_level", index=False)
            word_summary_df.to_excel(writer, sheet_name=f"{project}_word_summary", index=False)

        top_word_summary_df.to_excel(writer, sheet_name=f"Top_{TOP_N_WORDS}_word_summary", index=False)

    summary_lines = [
        "AI-associated word analysis completed.",
        "",
        f"Number of AI-associated words used: {len(AI_ASSOCIATED_WORDS)} words",
        f"Projects analyzed: {', '.join(PROJECTS)}",
        f"Years analyzed per project: {', '.join(YEARS)}",
        "",
        "Manuscript body handling:",
        "    The manuscript body was stored in three columns named:",
        "    Manuscript_text_1, Manuscript_text_2, and Manuscript_text_3.",
        "    These fields were concatenated in sequence to rebuild the full manuscript body",
        "    before total word counts and AI-associated word occurrences were calculated.",
        "",
        "Definition of combined density:",
        "    ((total AI-associated word occurrences in Title + Abstract + Manuscript body)",
        "     / total words in Title + Abstract + Manuscript body) * 10,000",
        "",
        "Definition of AI-associated word diversity ratio:",
        "    unique_AI_associated_words_used / number_of_AI_associated_words",
        "",
        "Word summary:",
        f"    The tabs 'Project_word_summary' reports the summary statistics for the AI-associated words",
        "    ranked by % relative change for each project.",
        "",
        f"Top {TOP_N_WORDS} word summary:",
        f"    The tab 'Top_{TOP_N_WORDS}_word_summary' reports the summary statistics (mean, median, minimum, and maximum)",
        "    for the top {TOP_N_WORDS} words by the % relative change for each project.",
        "",
        "Heatmap outputs:",
    ]

    for heatmap_path in heatmap_paths:
        summary_lines.append(f"    {heatmap_path.name}")

    summary_lines.extend([
        "",
        "Output sheets:"
    ])

    for project in PROJECTS:
        summary_lines.append(f"    {project}_manuscript_level")
        summary_lines.append(f"    {project}_word_summary")

    summary_lines.append(f"    Top_{TOP_N_WORDS}_word_summary")

    summary_txt = "\n".join(summary_lines)

    summary_path = OUTPUT_DIR / "README_summary.txt"
    summary_path.write_text(summary_txt, encoding="utf-8")

    print(f"Analyses saved to the results folder.")


if __name__ == "__main__":
    main()
