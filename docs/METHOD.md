# PhysicsGate method

For each target, the retained two-branch gate combines the direct prediction
and the physics-anchored residual prediction:

$$
\hat A_{gate}
= \hat A_D+w(\hat A_R-\hat A_D),
\quad
w = \frac{1}{1+e^{-g(R_1,R_2,R_3)}}.
$$

`g` is a standardized ridge regressor. Its training target is the clipped
sample-wise weight that minimizes squared error between the two OOF branches.
The fit is weighted by squared branch separation so that nearly identical
branches do not dominate the gate.

The published reliability features are:

1. `R1`, branch disagreement:
   `abs(A_direct - A_residual) / sd(y_train)`.
2. `R2`, physics-range risk: the amount by which `A_physics` lies outside the
   outer-training target range, normalized by that range.
3. `R3`, propagated uncertainty: the equation-propagation uncertainty proxy
   normalized by `sd(y_train)`.

No true validation or test auxiliary target is an input to a physics equation,
residual model, stacking model or gate.

The compact reference implementation is in `physicsgate/gate.py`. Physical
relationships and metric helpers are in `physicsgate/equations.py` and
`physicsgate/metrics.py`.
