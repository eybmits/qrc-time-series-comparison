# Student Guide

This repository shows one Quantum Reservoir Computing comparison:

```text
same quantum spin reservoir -> Mackey-Glass task
same quantum spin reservoir -> NARMA10 task
```

The goal is not to build the largest or fastest QRC model. The goal is to make
the memory mechanism easy to see, and to show that one reservoir can perform
differently on different time-series problems.

## The Big Idea

Reservoir computing has two parts:

```text
fixed dynamic system + trained linear readout
```

Here the fixed dynamic system is a quantum spin reservoir. The trained part is
only the final linear readout.

## The Two Tasks

### Mackey-Glass

Mackey-Glass is a delayed chaotic time series. The model sees the current value
and predicts the next value:

```text
x(t) -> predict x(t+1)
```

The default QRC predicts this very well.

### NARMA10

NARMA10 is a driven input-output memory task. The model sees an external input
signal and predicts the NARMA output:

```text
u(t) -> predict y(t+1)
```

This is harder for the same reservoir settings. That is the point of the
comparison.

## Where Memory Lives

In this code, the memory is the quantum reservoir state:

```python
self.rho
```

`rho` is a density matrix. It describes the current quantum state of the spin
system.

## One Time Step

At every time step, the code does this:

```text
1. Put the current input value into qubit 0.
2. Keep the old quantum state of the other qubits.
3. Evolve the whole quantum spin system.
4. Measure Pauli X and Z expectation values.
5. Use a linear readout to predict the next target value.
```

The key memory point is step 2. The other qubits are not reset.

## Why This Is Stateful

A stateless model forgets everything after each input.

This QRC model does not forget immediately, because the next quantum state uses
the previous quantum state:

```python
self.rho = self.unitary @ self.rho @ self.unitary_dagger
```

That line means:

```text
new quantum state = quantum evolution of old quantum state
```

So the reservoir carries information forward through time.

## What Gets Trained?

Only the final linear readout is trained.

The quantum spin reservoir is fixed after initialization. For the comparison,
both tasks use the same reservoir settings:

```python
N_QUBITS = 5
VIRTUAL_NODES = 7
EVOLUTION_TIME = 0.65
RIDGE = 1e-4
```

The readout is trained separately for each task because the targets are
different.

## What Result Should You See?

After running:

```bash
python qrc_time_series_comparison.py
```

you should see approximately:

```text
Mackey-Glass QRC RMSE: 0.005254
NARMA10 QRC RMSE:      0.649665
```

Both numbers are normalized RMSE values, so lower is better.

This shows the intended lesson: the same reservoir is excellent for
Mackey-Glass here, but not nearly as strong on NARMA10.

## What To Try Changing

Good exercises:

1. Set `N_QUBITS = 4`. Which task suffers more?
2. Set `VIRTUAL_NODES = 3`. Does NARMA10 get worse?
3. Increase `RIDGE`. Does either prediction become smoother?
4. Change `NARMA_ORDER` from `10` to another value and rerun.
5. Comment out the reservoir features and use only the current input. This
   should behave like the memoryless baseline.

