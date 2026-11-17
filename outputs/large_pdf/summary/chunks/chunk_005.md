### Summary

This section of the research paper discusses the results of experiments with a Transformer model applied to English-to-German translation and English constituency parsing tasks. The experiments are summarized in Table 3, which details the performance of the Transformer model on the development set of the English-to-German translation task. The key findings are:

1. **Translation Task (Table 3):**
   - **Model Variations:** The model was tested with varying parameters such as the number of attention heads, attention key and value dimensions, and dropout rate, while keeping the amount of computation constant.
   - **Key Observations:**
     - Single-head attention performed 0.9 BLEU worse than the best setting.
     - Reducing the attention key size (dk) negatively impacted model quality.
     - Smaller models performed worse than larger ones.
     - Dropout helped prevent overfitting.
     - Replacing sinusoidal positional embeddings with learned embeddings did not significantly affect performance.
   - **Best Model:** The best model had 4 layers, dmodel = 1024, dk = 16, dv = 16, and used 0.3 dropout.

2. **Parsing Task (Section 6.3):**
   - **Model Training:** The Transformer was trained on the Wall Street Journal (WSJ) portion of the Penn Treebank with 40K training sentences.
   - **Semi-supervised Setting:** The model was also trained using a larger corpus of approximately 17M sentences.
   - **Results:** The Transformer achieved a WSJ 23 F1 score of 91.3, outperforming previous state-of-the-art models including those by Vinyals &amp; Kaiser (2014), Petrov et al. (2006), Zhu et al. (2013), and Dyer et al. (2016).

In summary, the Transformer model demonstrated strong performance on both translation and constituency parsing tasks, with notable improvements in parsing due to the use of learned positional embeddings. The model's performance was robust across different parameter settings, with dropout being a key factor in preventing overfitting.
