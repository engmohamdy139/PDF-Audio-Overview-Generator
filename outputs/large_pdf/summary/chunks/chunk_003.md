### Summary

This section compares the computational complexity and architectural differences between self-attention, recurrent, and convolutional layers for mapping sequence data. The key points are:

1. **Self-Attention**:
   - Complexity: \(O(n^2 \cdot d)\)
   - Parallelization: \(O(1)\)
   - Path Length: \(O(1)\)
   - Positional Encoding: Uses sine and cosine functions of different frequencies to encode sequence positions.

2. **Recurrent Layers**:
   - Complexity: \(O(n \cdot d^2)\)
   - Parallelization: \(O(n)\)
   - Path Length: \(O(n)\)
   - Used for parallelizable operations within the sequence.

3. **Convolutional Layers**:
   - Complexity: \(O(k \cdot n \cdot d^2)\)
   - Parallelization: \(O(1)\)
   - Path Length: \(O(\log k(n))\)
   - Can be made more efficient with separable convolutions, reducing complexity to \(O(k \cdot n \cdot d + n \cdot d^2)\).

4. **Self-Attention (Restricted)**:
   - Complexity: \(O(r \cdot n \cdot d)\)
   - Parallelization: \(O(1)\)
   - Path Length: \(O(n/r)\)
   - Reduces maximum path length by considering only a neighborhood of size \(r\).

The section also discusses the desiderata for these layers, including computational complexity, parallelization, and the ability to learn long-range dependencies. Self-attention is found to be more efficient for shorter sequences, while convolutional layers are more efficient for longer sequences. The authors note that self-attention can be made more efficient with separable convolutions, and they plan to investigate further improvements.

### Key Findings

- Self-attention layers are more efficient for shorter sequences compared to recurrent and convolutional layers.
- Convolutional layers are more efficient for longer sequences but are generally more expensive.
- Self-attention can be optimized by considering only a neighborhood of positions around the output position.
- Separable convolutions can reduce the complexity of convolutional layers to match the complexity of self-attention layers.
- Self-attention can potentially yield more interpretable models by learning different tasks and exhibiting syntactic and semantic structure behavior.

### Limitations

- The choice of positional encoding (sine and cosine functions) was used for simplicity and theoretical considerations.
- The effectiveness of learned positional embeddings was not explored in this work.
- The impact of different hardware configurations on training performance was not detailed.
