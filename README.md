# QRC Time-Series Comparison

This is a deliberately small teaching repository for **Quantum Reservoir
Computing (QRC)**.

It compares the same quantum spin reservoir on two time-series tasks:

1. Mackey-Glass next-step prediction
2. NARMA10 input-output prediction

The main lesson is that the **same reservoir settings can perform very
differently on different tasks**. In this default example, the reservoir works
very well on Mackey-Glass and less well on NARMA10.

The code uses only NumPy and Matplotlib. There is no quantum SDK dependency,
because the goal is to make the mechanics readable.

## Files

- `qrc_time_series_comparison.py` - the full example, written to be read top to bottom
- `STUDENT_GUIDE.md` - a short plain-language walkthrough for first reading
- `requirements.txt` - minimal Python dependencies
- `results/` - example output created by running the script

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python qrc_time_series_comparison.py
```

The script prints RMSE values and saves a comparison plot to:

```text
results/qrc_time_series_comparison.png
```

## Suggested Reading Order

1. Read `STUDENT_GUIDE.md`.
2. Run `python qrc_time_series_comparison.py`.
3. Open `results/qrc_time_series_comparison.png`.
4. Read only these parts of the Python file first:
   - `make_mackey_glass`
   - `make_narma`
   - `QuantumSpinReservoir.inject_input`
   - `QuantumSpinReservoir.evolve_and_measure`
   - `run_qrc_task`

## What Makes This QRC?

The reservoir is a small quantum spin system with a fixed Hamiltonian:

```python
rho(t + dt) = U rho(t) U_dagger
```

At each time step:

1. The current input value is encoded into qubit 0.
2. The other qubits keep their reduced quantum state.
3. The whole spin system evolves under a fixed Hamiltonian.
4. Pauli `X` and `Z` expectation values are measured.
5. A linear readout predicts the target at the next time step.

Only the linear readout is trained. The quantum reservoir itself is fixed after
random initialization.

## Where Memory Lives

The key object is:

```python
self.rho
```

`self.rho` is the quantum density matrix of the reservoir. It is carried from
one time step to the next. It is not reset inside the prediction loop. Because
every new quantum state depends on the previous quantum state, the reservoir
contains memory of the recent input history.

The key state update is:

```python
self.rho = self.unitary @ self.rho @ self.unitary_dagger
```

The input injection is:

```python
self.rho = np.kron(input_rho, memory_rho)
```

Here `input_rho` is the new input qubit. `memory_rho` is the reduced density
matrix of the other qubits. Those other qubits are the memory part of the
reservoir.

## Expected Result

With the default seed and settings, the example should produce approximately:

```text
Mackey-Glass:
- QRC one-step RMSE, normalized scale:       0.005254
- Memoryless one-step baseline RMSE:         0.146814

NARMA10:
- QRC one-step RMSE, normalized scale:       0.649665
- Memoryless one-step baseline RMSE:         0.836093
```

Small numerical differences are normal across machines.

## Teaching Notes

- The same QRC settings are used for both tasks.
- Each task starts from a fresh copy of the same reservoir, with the same
  Hamiltonian seed.
- The readout is trained separately for each task.
- Mackey-Glass is a self-prediction task: input and target are the same signal.
- NARMA10 is an input-output task: the reservoir sees `u(t)` and predicts the
  generated target `y(t+1)`.
- The NARMA10 score is worse here, which is useful pedagogically: reservoir
  quality is task-dependent.
- `N_QUBITS` controls the quantum reservoir size.
- `VIRTUAL_NODES` controls how many intermediate measurements are collected per input.
- `EVOLUTION_TIME` controls how long the quantum system evolves between virtual nodes.
- `RIDGE` regularizes the final linear readout.

