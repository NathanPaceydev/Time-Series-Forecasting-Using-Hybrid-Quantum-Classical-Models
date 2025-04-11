
---
First get working hybrid
Then compare to my thesis 
maybe add more data 
build classic ML
compare hyper parameter's and setup to thesis
iterate

---
Issues encountered
- data normalization / window on stock options is a complex issue
- ![alt text](image.png)
    The quantum circuit was making predictions in a very tight band, indicating the quantum layer isn't learning meaningful transformations

    lstm.weight_ih_l0 grad mean: 0.000000
lstm.weight_hh_l0 grad mean: 0.000000
Component	Behavior
quantum_out	Always [0.9999272, 0.99993163] — totally static
final output	Only slightly changes (depends on linear2)
gradients	All zero in lstm and linear1 layers
only linear2	Receives gradient and updates (after quantum layer)
🧠 What this tells us:
The quantum layer is completely ignoring its inputs

So: the issue is gradient flow is broken between your loss and your quantum circuit.

Fixed this through the function calll for inputs the data type of inputs and the type of activation function
this combination ReLu especially was forcing data to zero

-- 
Decided to create visuals for the Quantum circuit and the overall model
added dynamic logging of design and results to expirement and compare
Idea to make the model fully customizable like Terra's platform and dynamically log results 

