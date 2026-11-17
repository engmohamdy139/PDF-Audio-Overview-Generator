### Summary of Section 5.4: Regularization

**Key Ideas:**
- The Transformer model achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests.
- The Transformer model outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU points on the English-to-German task, achieving a new state-of-the-art BLEU score of 28.4.
- The Transformer model outperforms all previously published single models on the English-to-French task, achieving a BLEU score of 41.0, at less than 1/4 the training cost of the previous state-of-the-art model.
- Three types of regularization are employed during training: residual dropout, label smoothing, and a fraction of the training cost.

**Technical Details:**
- **Residual Dropout:** Dropout is applied to the output of each sub-layer before it is added to the sub-layer input and normalized. Embeddings and positional encodings are also subject to dropout. For the base model, the dropout rate is set to Pdrop = 0.1.
- **Label Smoothing:** Label smoothing with a value of ϵls = 0.1 is used during training. This technique helps the model learn to be more uncertain, which improves accuracy and BLEU score.
- **Training Cost:** The training cost is estimated by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU.

**Findings:**
- The Transformer model significantly outperforms previous models in BLEU scores, particularly on the English-to-French task, with a substantial reduction in training cost.
