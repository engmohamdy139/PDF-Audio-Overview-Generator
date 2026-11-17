### Summary

This section discusses the results of experiments using the Transformer model, a sequence transduction model based on attention mechanisms. The experiments were conducted on translation tasks, specifically on the WMT 2014 English-to-German and English-to-French translation datasets. The Transformer model achieved state-of-the-art results, outperforming previous models including the Berkeley Parser and RNN sequence-to-sequence models. 

Key findings include:
- The Transformer model outperformed the Berkeley Parser even when trained only on the WSJ training set of 40K sentences.
- The Transformer model achieved a new state-of-the-art performance on both WMT 2014 English-to-German and English-to-French translation tasks.
- The Transformer model performed better than all previously reported models, except for the Recurrent Neural Network Grammar model, in both WSJ only and semi-supervised settings.
- The Transformer model achieved results of 91.3% and 92.1% in semi-supervised and multi-task settings, respectively, and 93.0% in a multi-task setting using the Transformer model (4 layers).
- The Transformer model achieved 93.3% in a generative setting.

The Transformer model outperformed RNN sequence-to-sequence models in terms of speed and performance. The experiments used a beam size of 21 and α = 0.3 for both WSJ only and semi-supervised settings. The maximum output length was increased to input length + 300.
