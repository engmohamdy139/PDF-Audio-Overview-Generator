### Summary of Section 3.3: Position-wise Feed-Forward Networks

**Key Ideas:**
- Each layer in the encoder and decoder contains a position-wise feed-forward network (FFN).
- The FFN consists of two linear transformations with a ReLU activation function in between.
- The FFN is applied to each position separately and identically, with different parameters for each layer.
- The FFN has a dimensionality of dmodel = 512, and an inner-layer dimensionality of df_f = 2048.

**Technical Details:**
- The FFN is defined as: FFN(x) = max(0, xW1 + b1)W2 + b2.
- The FFN uses the same linear transformations across different positions but different parameters for each layer.
- The FFN can be described as two convolutions with a kernel size of 1.

### Embeddings and Softmax

**Key Ideas:**
- The model uses learned embeddings to convert input and output tokens into vectors of dimension dmodel.
- A pre-softmax linear transformation and softmax function are used to convert decoder output to predicted next-token probabilities.
- The same weight matrix is shared between embedding layers and the pre-softmax linear transformation.
- Weights are multiplied by √dmodel.

### Complexity and Sequential Operations

**Key Ideas:**
- Different types of layers have varying complexities and sequential operations.
- Self-Attention layers have complexity O(n²·d) and sequential operations O(1).
- Recurrent layers have complexity O(n·d²) and sequential operations O(n).
- Convolutional layers have complexity O(k·n·d²) and sequential operations O(1).
- Self-Attention (restricted) layers have complexity O(r·n·d) and sequential operations O(1).

### Positional Encoding

**Key Ideas:**
- To make use of the order of the sequence, positional encodings are added to the input embeddings.
- Positional encodings are sinusoidal functions of different frequencies, where each dimension corresponds to a sinusoid.
- The positional encodings are geometrically spaced, with wavelengths ranging from 2π to 10000·2π.
- The model uses sine and cosine functions to create the positional encodings.
- The sinusoidal version of positional encodings was found to produce nearly identical results to learned positional embeddings, but may allow extrapolation to longer sequence lengths.
