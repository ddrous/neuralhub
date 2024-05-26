
## The various results

- SGD: Simple SGD, with T from 1 to 25
- ADAM: Adam, results in more vertical event `horizon` fronts (EH fronts).
- SGD 2: Smaller time horizons considered, from 2 to 16
- SGD 3: 
- SGD 4: Observes the gradients of the norms rescaled by `T`
- SGD 5: Tests the minimax hypothesis. Minimize along the model, but maximize the loss along the final time T. It used the `main_renormalised` script. Best results, best EH fronts.


The term we introduce is the front: The event horizon fronts, the point after which a neural ODE doesn't return.
