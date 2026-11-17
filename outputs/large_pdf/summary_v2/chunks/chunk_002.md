### Summary of the Attention Section

**Key Ideas:**
- **Attention Function:** Maps a query and a set of key-value pairs to an output, using a compatibility function weighted by the dot product of the query and keys.
- **Scaled Dot-Product Attention:** Computes the dot products of queries with keys, divides by √ dk, and applies a softmax function to compute the weights on values. This is used in practice due to its speed and space efficiency.
- **Multi-Head Attention:** Projects queries, keys, and values into multiple heads, each performing attention independently. The outputs from these heads are concatenated and then projected back to the original dimension. This allows the model to jointly attend to information from different subspaces at different positions.
- **Applications:** Used in "encoder-decoder attention" layers, self-attention layers in the encoder and decoder, and includes mechanisms to prevent leftward information flow in the decoder.

**Technical Details:**
- **Scaled Dot-Product Attention:** Uses a scaling factor of √ 1/dk to prevent the softmax function from producing extremely small gradients for large dot products.
- **Multi-Head Attention:** Employs h parallel attention layers, each with dk = dv = dmodel/h dimensions, to reduce the computational cost while maintaining performance.

**Findings:**
- Scaled Dot-Product Attention is more efficient and effective in practice compared to additive attention, especially for larger dk values.
- Multi-Head Attention enhances the model's ability to attend to different subspaces, improving its performance on tasks requiring such multi-dimensional attention.
