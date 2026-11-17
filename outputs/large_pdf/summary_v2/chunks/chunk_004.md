### Summary of Research Paper Section

This section discusses the use of self-attention layers in comparison to recurrent and convolutional layers for mapping sequences. The key desiderata considered are:

1. **Computational Complexity**: Self-attention layers are faster than recurrent layers when the sequence length is smaller than the representation dimensionality, making them more efficient for tasks involving long sequences.
2. **Parallelization**: Self-attention layers have a constant number of sequentially executed operations, allowing for higher parallelization compared to recurrent layers.
3. **Path Length**: Self-attention layers have shorter maximum path lengths between input and output positions, facilitating the learning of long-range dependencies in sequences.

The section also mentions that self-attention could be restricted to considering a neighborhood of size \( r \) in the input sequence, increasing the maximum path length to \( O(n/r) \). This approach could be explored further in future work.

### Training Details

- **Training Data**: The models were trained on the WMT 2014 English-German dataset (4.5 million sentence pairs) and the WMT 2014 English-French dataset (36 million sentence pairs). Sentence encoding used byte-pair encoding or word-piece encoding.
- **Batching**: Sentences were batched by approximate sequence length, with each batch containing approximately 25,000 source and target tokens.
- **Hardware and Schedule**: Training was performed on one machine with 8 NVIDIA P100 GPUs. The base models were trained for 100,000 steps (12 hours) and the big models for 300,000 steps (3.5 days).
- **Optimizer**: The Adam optimizer was used with hyperparameters β1 = 0.9, β2 = 0.98, and ϵ = 10^-9. The learning rate varied over the course of training, increasing linearly for the first 4,000 steps and decreasing thereafter proportionally to the inverse square root of the step number.
