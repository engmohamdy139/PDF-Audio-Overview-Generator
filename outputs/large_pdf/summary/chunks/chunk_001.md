### Summary

The provided section discusses the Transformer model architecture, focusing on its encoder and decoder stacks, and the attention mechanism used within these components. The key points are as follows:

1. **Encoder and Decoder Stacks**:
   - The encoder and decoder both consist of a stack of 6 identical layers.
   - Each layer contains two sub-layers: a multi-head self-attention mechanism and a position-wise, fully connected feed-forward network.
   - Residual connections are used around each sub-layer, followed by layer normalization.
   - The embedding layers also produce outputs of dimension 512.

2. **Attention Mechanism**:
   - The attention mechanism maps a query and a set of key-value pairs to an output.
   - The output is computed as a weighted sum of the values, where weights are assigned based on a compatibility function of the query with the corresponding key.
   - Two common attention functions are additive attention and dot-product (multiplicative) attention.
   - The Transformer uses a scaled dot-product attention mechanism, which involves dividing the dot product of query and key by √dk and applying a softmax function to the result.

3. **Multi-Head Attention**:
   - Multi-Head Attention consists of several attention layers running in parallel.
   - The attention mechanism in the decoder is modified to prevent positions from attending to subsequent positions, using a masking technique.

The Transformer model relies entirely on self-attention to compute representations of its input and output, unlike previous models which used sequence-aligned recurrent networks or convolution. This architecture allows for efficient computation of dependencies between distant positions, though it reduces effective resolution due to averaging attention-weighted positions.
