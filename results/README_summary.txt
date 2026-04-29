AI-associated word analysis completed.

Number of AI-associated words used: 50 words
Projects analyzed: JAPPL, NIH
Years analyzed per project: 2020, 2026

Manuscript body handling:
    The manuscript body was stored in three columns named:
    Manuscript_text_1, Manuscript_text_2, and Manuscript_text_3.
    These fields were concatenated in sequence to rebuild the full manuscript body
    before total word counts and AI-associated word occurrences were calculated.

Definition of combined density:
    ((total AI-associated word occurrences in Title + Abstract + Manuscript body)
     / total words in Title + Abstract + Manuscript body) * 10,000

Definition of AI-associated word diversity ratio:
    unique_AI_associated_words_used / number_of_AI_associated_words

Word summary:
    The tabs 'Project_word_summary' reports the summary statistics for the AI-associated words
    ranked by % relative change for each project.

Top 20 word summary:
    The tab 'Top_20_word_summary' reports the summary statistics (mean, median, minimum, and maximum)
    for the top {TOP_N_WORDS} words by the % relative change for each project.

Heatmap outputs:
    JAPPL_density_heatmap_top_20_words.png
    NIH_density_heatmap_top_20_words.png

Output sheets:
    JAPPL_manuscript_level
    JAPPL_word_summary
    NIH_manuscript_level
    NIH_word_summary
    Top_20_word_summary