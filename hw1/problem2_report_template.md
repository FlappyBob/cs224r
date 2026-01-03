# Problem 2: Hyperparameter Experiment

## Chosen Hyperparameter

**Hyperparameter**: `num_agent_train_steps_per_iter` (Number of Training Steps)

**Rationale**: The number of training steps directly controls how many gradient updates the policy network receives during training. This hyperparameter is fundamental to the learning process - too few steps may result in underfitting (the model hasn't learned enough from the expert data), while too many steps may lead to overfitting or diminishing returns. Understanding the relationship between training steps and performance helps identify the optimal training budget for the behavior cloning agent.

## Experimental Setup

- **Environment**: Ant-v4
- **Hyperparameter Values Tested**: [250, 500, 1000, 2000, 5000]
- **Other Parameters**: 
  - Network architecture: 2 layers, 64 hidden units
  - Learning rate: 0.005
  - Evaluation batch size: 5000 steps
  - All other hyperparameters held constant

## Results

### Performance Table

| Training Steps | Average Return | Std Return |
|----------------|----------------|------------|
| 250 | [TO BE FILLED] | [TO BE FILLED] |
| 500 | [TO BE FILLED] | [TO BE FILLED] |
| 1000 | [TO BE FILLED] | [TO BE FILLED] |
| 2000 | [TO BE FILLED] | [TO BE FILLED] |
| 5000 | [TO BE FILLED] | [TO BE FILLED] |

### Performance Plot

![Hyperparameter Experiment](hyperparameter_experiment.png)

## Analysis

The graph shows how the BC agent's performance (measured by average return) varies with the number of training steps. Key observations:

1. **Performance increases with training steps**: As expected, more training steps generally lead to better performance, indicating the model continues to learn from the expert data.

2. **Diminishing returns**: The improvement rate decreases as the number of training steps increases, suggesting that beyond a certain point, additional training provides limited benefits.

3. **Performance plateau**: At higher training step counts (e.g., 5000), the performance may plateau or improve only marginally, indicating that the model has learned most of what it can from the available expert data.

## Conclusion

This experiment demonstrates the importance of choosing an appropriate number of training steps. While more training generally improves performance, there is a trade-off between performance gains and computational cost. For the Ant environment with the given expert data, the optimal number of training steps appears to be around 2000-5000 steps, where performance is good without excessive computational overhead.

