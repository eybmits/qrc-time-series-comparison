"""
Introductory time-series comparison with a quantum spin reservoir.

Run:
    python qrc_time_series_comparison.py

This is a deliberately small Quantum Reservoir Computing (QRC) example.

It runs the same quantum spin reservoir on two benchmark tasks:

1. Mackey-Glass next-step prediction.
2. NARMA10 input-output prediction.

The point of the comparison is simple:

    The same reservoir can work very well on one task and less well on another.

What makes it a QRC example?

1. The reservoir is a small quantum spin system.
2. Its state is a density matrix, called `rho` in the code.
3. The current scalar input is encoded into one input qubit.
4. The remaining qubits keep their previous quantum state.
5. The spin system evolves with a fixed Hamiltonian.
6. Only the final linear readout is trained.

The most important teaching point:

    The reservoir memory is the density matrix `rho`.

We do not reset `rho` inside the time loop. Each new state depends on the
previous state, so the reservoir carries information forward in time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Easy-to-change teaching settings
# ---------------------------------------------------------------------------

SEED = 3

TRAIN_STEPS = 2500
TEST_STEPS = 1200
SHORT_FREE_RUN_STEPS = 20
WASHOUT_STEPS = 200

N_QUBITS = 5
VIRTUAL_NODES = 7
EVOLUTION_TIME = 0.65
RIDGE = 1e-4

NARMA_ORDER = 10


# ---------------------------------------------------------------------------
# Basic quantum objects
# ---------------------------------------------------------------------------

COMPLEX = np.complex128

I2 = np.eye(2, dtype=COMPLEX)
X = np.array([[0, 1], [1, 0]], dtype=COMPLEX)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX)


def kron_all(operators):
    """Kronecker product of a list of single-qubit operators."""

    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def one_qubit_operator(n_qubits, qubit, operator):
    """Place a one-qubit operator on one qubit of an n-qubit system."""

    operators = [I2] * n_qubits
    operators[qubit] = operator
    return kron_all(operators)


def two_qubit_operator(n_qubits, qubit_a, operator_a, qubit_b, operator_b):
    """Place two one-qubit operators on two different qubits."""

    operators = [I2] * n_qubits
    operators[qubit_a] = operator_a
    operators[qubit_b] = operator_b
    return kron_all(operators)


def unitary_from_hamiltonian(hamiltonian, time_step):
    """Compute U = exp(-i H dt) using NumPy diagonalization."""

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * eigenvalues * time_step)
    return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T


def expectation(rho, operator):
    """Expectation value Tr(rho O), returned as a real number."""

    return float(np.real(np.trace(rho @ operator)))


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------


def make_mackey_glass(
    n_points,
    tau=17,
    beta=0.2,
    gamma=0.1,
    power=10,
    dt=1.0,
    seed=7,
):
    """Generate a Mackey-Glass time series with a simple Euler update.

    The continuous equation is:

        dx/dt = beta * x(t - tau) / (1 + x(t - tau)^power) - gamma * x(t)

    The equation uses the delayed value x(t - tau), so the task rewards models
    that can keep useful history in their internal state.
    """

    rng = np.random.default_rng(seed)

    total_length = n_points + tau + 1
    x = np.empty(total_length)

    # A short initial history. The tiny noise avoids a perfectly flat start.
    x[: tau + 1] = 1.2 + 0.01 * rng.normal(size=tau + 1)

    for t in range(tau, total_length - 1):
        delayed_x = x[t - tau]
        dx = beta * delayed_x / (1.0 + delayed_x**power) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx

    return x[tau + 1 :]


def make_narma(n_points, order=NARMA_ORDER, seed=23):
    """Generate a standard NARMA input-output benchmark.

    NARMA is not a self-prediction task. The reservoir receives the external
    input u(t), and the target is the NARMA output y(t).

    For NARMA10, the next output depends on the current output, a sum of recent
    outputs, and a product of delayed input values. This is a harder memory and
    nonlinearity test than the Mackey-Glass one-step task in this example.
    """

    rng = np.random.default_rng(seed)
    inputs = rng.uniform(0.0, 0.5, size=n_points)
    outputs = np.zeros(n_points)

    for t in range(order - 1, n_points - 1):
        recent_output_sum = np.sum(outputs[t - order + 1 : t + 1])
        outputs[t + 1] = (
            0.3 * outputs[t]
            + 0.05 * outputs[t] * recent_output_sum
            + 1.5 * inputs[t - order + 1] * inputs[t]
            + 0.1
        )

    return inputs, outputs


# ---------------------------------------------------------------------------
# Quantum spin reservoir
# ---------------------------------------------------------------------------


class QuantumSpinReservoir:
    """A small stateful quantum spin reservoir.

    The reservoir state is a density matrix:

        self.rho

    `self.rho` is the memory. It is updated at every time step and carried into
    the next time step.
    """

    def __init__(self, n_qubits, virtual_nodes, evolution_time, seed):
        self.n_qubits = n_qubits
        self.virtual_nodes = virtual_nodes

        self.hamiltonian = self._make_spin_hamiltonian(seed)
        self.unitary = unitary_from_hamiltonian(self.hamiltonian, evolution_time)
        self.unitary_dagger = self.unitary.conj().T

        dimension = 2**n_qubits

        # Start from a neutral mixed state. After the washout period, the exact
        # initial state is no longer important.
        self.rho = np.eye(dimension, dtype=COMPLEX) / dimension

        # We read out simple Pauli observables from every qubit. These are the
        # measured reservoir features passed to the linear model.
        self.readout_operators = []
        for qubit in range(n_qubits):
            self.readout_operators.append(one_qubit_operator(n_qubits, qubit, Z))
            self.readout_operators.append(one_qubit_operator(n_qubits, qubit, X))

    def _make_spin_hamiltonian(self, seed):
        """Create a fixed disordered quantum spin Hamiltonian.

        This is a small transverse-field spin model with nearest-neighbor and
        weak longer-range couplings. The Hamiltonian is random but fixed by the
        seed, so the example is reproducible.
        """

        rng = np.random.default_rng(seed)
        dimension = 2**self.n_qubits
        hamiltonian = np.zeros((dimension, dimension), dtype=COMPLEX)

        # Local fields.
        for qubit in range(self.n_qubits):
            hamiltonian += rng.uniform(0.3, 1.1) * one_qubit_operator(
                self.n_qubits, qubit, X
            )
            hamiltonian += rng.uniform(-0.25, 0.25) * one_qubit_operator(
                self.n_qubits, qubit, Z
            )

        # Nearest-neighbor spin-spin couplings.
        for qubit in range(self.n_qubits - 1):
            hamiltonian += rng.uniform(0.4, 1.2) * two_qubit_operator(
                self.n_qubits, qubit, Z, qubit + 1, Z
            )
            hamiltonian += rng.uniform(-0.7, 0.7) * two_qubit_operator(
                self.n_qubits, qubit, X, qubit + 1, X
            )

        # Weak longer-range couplings make the dynamics richer.
        for qubit_a in range(self.n_qubits):
            for qubit_b in range(qubit_a + 2, self.n_qubits):
                hamiltonian += rng.uniform(-0.25, 0.25) * two_qubit_operator(
                    self.n_qubits, qubit_a, Z, qubit_b, Z
                )

        return hamiltonian

    def _trace_out_input_qubit(self):
        """Return the reduced density matrix of all qubits except qubit 0."""

        rest_dimension = 2 ** (self.n_qubits - 1)
        rho_reshaped = self.rho.reshape(2, rest_dimension, 2, rest_dimension)

        # Partial trace over the first qubit:
        # sum over <0|rho|0> and <1|rho|1>.
        return rho_reshaped[0, :, 0, :] + rho_reshaped[1, :, 1, :]

    def _input_qubit_state(self, input_value_01):
        """Encode a scalar in [0, 1] as a one-qubit pure state."""

        input_value_01 = float(np.clip(input_value_01, 0.0, 1.0))

        ket = np.array(
            [
                np.sqrt(1.0 - input_value_01),
                np.sqrt(input_value_01),
            ],
            dtype=COMPLEX,
        )

        return np.outer(ket, ket.conj())

    def inject_input(self, input_value_01):
        """Inject the current input into qubit 0.

        This is the standard QRC input step used here:

        1. Replace qubit 0 by a new input state.
        2. Keep the reduced state of the remaining qubits.

        The remaining qubits are not reset. They carry the reservoir memory.
        """

        input_rho = self._input_qubit_state(input_value_01)
        memory_rho = self._trace_out_input_qubit()
        self.rho = np.kron(input_rho, memory_rho)

    def evolve_and_measure(self):
        """Evolve the quantum state and collect virtual-node measurements."""

        features = []

        for _ in range(self.virtual_nodes):
            # This is the stateful quantum evolution:
            # rho(t + dt) = U rho(t) U_dagger
            self.rho = self.unitary @ self.rho @ self.unitary_dagger

            for operator in self.readout_operators:
                features.append(expectation(self.rho, operator))

        return np.array(features)

    def step(self, input_value_01):
        """Inject one input, evolve, measure, and keep the new quantum state."""

        self.inject_input(input_value_01)
        return self.evolve_and_measure()

    def get_state(self):
        """Save the current density matrix."""

        return self.rho.copy()

    def set_state(self, rho):
        """Restore a saved density matrix."""

        self.rho = rho.copy()


# ---------------------------------------------------------------------------
# Learning utilities
# ---------------------------------------------------------------------------


def readout_features(current_input_z, qrc_measurements):
    """Combine bias, current input, and QRC memory measurements."""

    return np.concatenate(([1.0, current_input_z], qrc_measurements))


def fit_ridge_regression(features, targets, ridge):
    """Fit a linear readout with closed-form ridge regression."""

    identity = np.eye(features.shape[1])
    return np.linalg.solve(
        features.T @ features + ridge * identity,
        features.T @ targets,
    )


def rmse(predictions, targets):
    """Root mean squared error."""

    return np.sqrt(np.mean((predictions - targets) ** 2))


def scale_to_unit_interval(values, train_min, train_max):
    """Map values to [0, 1] for input injection."""

    if np.isclose(train_min, train_max):
        return np.zeros_like(values)

    return np.clip((values - train_min) / (train_max - train_min), 0.0, 1.0)


def z_score(values, train_mean, train_std):
    """Normalize values using training statistics."""

    if np.isclose(train_std, 0.0):
        raise ValueError("Cannot z-score a constant training signal.")

    return (values - train_mean) / train_std


def denormalize(values_z, mean, std):
    """Map normalized values back to the original scale."""

    return values_z * std + mean


def prepare_task_signals(input_signal, target_signal):
    """Prepare input and target arrays without leaking test statistics."""

    train_input = input_signal[:TRAIN_STEPS]
    train_target = target_signal[:TRAIN_STEPS]

    input_mean = train_input.mean()
    input_std = train_input.std()
    target_mean = train_target.mean()
    target_std = train_target.std()

    input_z = z_score(input_signal, input_mean, input_std)
    target_z = z_score(target_signal, target_mean, target_std)

    input_min = train_input.min()
    input_max = train_input.max()
    input_values_01 = scale_to_unit_interval(input_signal, input_min, input_max)

    return {
        "input_z": input_z,
        "target_z": target_z,
        "input_values_01": input_values_01,
        "input_mean": input_mean,
        "input_std": input_std,
        "input_min": input_min,
        "input_max": input_max,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def train_qrc(input_z, target_z, input_values_01):
    """Run the training sequence once and fit the linear readout."""

    reservoir = QuantumSpinReservoir(
        n_qubits=N_QUBITS,
        virtual_nodes=VIRTUAL_NODES,
        evolution_time=EVOLUTION_TIME,
        seed=SEED,
    )

    collected_features = []
    targets = []

    for t in range(TRAIN_STEPS):
        qrc_measurements = reservoir.step(input_values_01[t])

        # Washout lets the initially neutral quantum state synchronize with the
        # signal before we ask the readout to learn from it.
        if t >= WASHOUT_STEPS:
            collected_features.append(readout_features(input_z[t], qrc_measurements))
            targets.append(target_z[t + 1])

    features = np.vstack(collected_features)
    targets = np.array(targets)
    readout_weights = fit_ridge_regression(features, targets, RIDGE)

    return reservoir, readout_weights


def predict_one_step(reservoir, readout_weights, input_z, input_values_01):
    """Teacher-forced prediction: the true current input is always supplied."""

    predictions = []

    for t in range(TRAIN_STEPS, TRAIN_STEPS + TEST_STEPS):
        qrc_measurements = reservoir.step(input_values_01[t])
        features = readout_features(input_z[t], qrc_measurements)
        predictions.append(features @ readout_weights)

    return np.array(predictions)


def predict_short_self_feedback_run(
    reservoir,
    readout_weights,
    first_input_z,
    train_z_min,
    train_z_max,
    input_mean,
    input_std,
    input_min,
    input_max,
):
    """Short autonomous rollout for self-prediction tasks.

    This is used for Mackey-Glass, where the input signal and target signal are
    the same time series. It is not used for NARMA, because NARMA is driven by
    an external random input sequence.

    We clip the feedback to the training range before reinjecting it. That keeps
    the demonstration numerically well behaved and easy to inspect.
    """

    predictions = []
    current_value_z = first_input_z

    for _ in range(SHORT_FREE_RUN_STEPS):
        clipped_value_z = float(np.clip(current_value_z, train_z_min, train_z_max))
        clipped_value = denormalize(clipped_value_z, input_mean, input_std)
        input_value_01 = scale_to_unit_interval(clipped_value, input_min, input_max)

        qrc_measurements = reservoir.step(input_value_01)
        features = readout_features(clipped_value_z, qrc_measurements)
        prediction_z = features @ readout_weights
        predictions.append(prediction_z)

        # This line makes the prediction autonomous.
        current_value_z = prediction_z

    return np.array(predictions)


def fit_memoryless_baseline(input_z, target_z):
    """Fit target(t+1) from only input(t), with no reservoir state."""

    features = []
    targets = []

    for t in range(WASHOUT_STEPS, TRAIN_STEPS):
        features.append([1.0, input_z[t]])
        targets.append(target_z[t + 1])

    features = np.array(features)
    targets = np.array(targets)
    return fit_ridge_regression(features, targets, RIDGE)


def run_qrc_task(task_name, input_signal, target_signal, include_short_free_run):
    """Train and evaluate the same QRC architecture on one task."""

    prepared = prepare_task_signals(input_signal, target_signal)

    reservoir, readout_weights = train_qrc(
        prepared["input_z"],
        prepared["target_z"],
        prepared["input_values_01"],
    )
    state_after_training = reservoir.get_state()

    predictions_z = predict_one_step(
        reservoir,
        readout_weights,
        prepared["input_z"],
        prepared["input_values_01"],
    )
    targets_z = prepared["target_z"][TRAIN_STEPS + 1 : TRAIN_STEPS + TEST_STEPS + 1]

    baseline_weights = fit_memoryless_baseline(
        prepared["input_z"],
        prepared["target_z"],
    )
    baseline_predictions_z = np.array(
        [
            np.array([1.0, prepared["input_z"][t]]) @ baseline_weights
            for t in range(TRAIN_STEPS, TRAIN_STEPS + TEST_STEPS)
        ]
    )

    predictions = denormalize(
        predictions_z,
        prepared["target_mean"],
        prepared["target_std"],
    )
    targets = denormalize(
        targets_z,
        prepared["target_mean"],
        prepared["target_std"],
    )

    result = {
        "task_name": task_name,
        "predictions_z": predictions_z,
        "targets_z": targets_z,
        "predictions": predictions,
        "targets": targets,
        "qrc_rmse_z": rmse(predictions_z, targets_z),
        "qrc_rmse": rmse(predictions, targets),
        "baseline_rmse_z": rmse(baseline_predictions_z, targets_z),
        "reservoir_shape": reservoir.rho.shape,
        "short_free_run_rmse_z": None,
        "short_free_run_rmse": None,
        "short_free_run_predictions": None,
        "short_free_run_targets": None,
    }

    if include_short_free_run:
        reservoir.set_state(state_after_training)
        free_predictions_z = predict_short_self_feedback_run(
            reservoir,
            readout_weights,
            first_input_z=prepared["input_z"][TRAIN_STEPS],
            train_z_min=prepared["input_z"][:TRAIN_STEPS].min(),
            train_z_max=prepared["input_z"][:TRAIN_STEPS].max(),
            input_mean=prepared["input_mean"],
            input_std=prepared["input_std"],
            input_min=prepared["input_min"],
            input_max=prepared["input_max"],
        )
        free_targets_z = prepared["target_z"][
            TRAIN_STEPS + 1 : TRAIN_STEPS + SHORT_FREE_RUN_STEPS + 1
        ]

        free_predictions = denormalize(
            free_predictions_z,
            prepared["target_mean"],
            prepared["target_std"],
        )
        free_targets = denormalize(
            free_targets_z,
            prepared["target_mean"],
            prepared["target_std"],
        )

        result["short_free_run_rmse_z"] = rmse(free_predictions_z, free_targets_z)
        result["short_free_run_rmse"] = rmse(free_predictions, free_targets)
        result["short_free_run_predictions"] = free_predictions
        result["short_free_run_targets"] = free_targets

    return result


def make_comparison_plot(results, output_path):
    """Save a plot comparing QRC predictions on both tasks."""

    n_to_show = 300

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 7), sharex=False)

    for axis, result in zip(axes, results):
        axis.plot(
            result["targets"][:n_to_show],
            label="true target",
            linewidth=2,
        )
        axis.plot(
            result["predictions"][:n_to_show],
            label="QRC prediction",
            linestyle="--",
        )
        axis.set_title(
            f"{result['task_name']}: QRC RMSE {result['qrc_rmse_z']:.4f}, "
            f"memoryless baseline {result['baseline_rmse_z']:.4f}"
        )
        axis.set_ylabel("target")
        axis.legend()

    axes[-1].set_xlabel("test step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_summary(results, plot_path):
    """Build the printed and saved text summary."""

    lines = [
        "Quantum spin reservoir time-series comparison",
        "",
        "Same QRC settings for both tasks:",
        f"- {N_QUBITS} qubits",
        f"- {VIRTUAL_NODES} virtual measurement nodes per input",
        f"- evolution time {EVOLUTION_TIME}",
        f"- ridge {RIDGE}",
        f"- Hamiltonian seed {SEED}",
        f"- density-matrix state rho with shape {results[0]['reservoir_shape']}",
        "",
        "Task results:",
    ]

    for result in results:
        lines.extend(
            [
                "",
                f"{result['task_name']}:",
                f"- QRC one-step RMSE, normalized scale:       "
                f"{result['qrc_rmse_z']:.6f}",
                f"- QRC one-step RMSE, original scale:         "
                f"{result['qrc_rmse']:.6f}",
                f"- Memoryless one-step baseline RMSE:         "
                f"{result['baseline_rmse_z']:.6f}",
            ]
        )

        if result["short_free_run_rmse_z"] is not None:
            lines.extend(
                [
                    f"- Short free-run QRC RMSE, normalized scale: "
                    f"{result['short_free_run_rmse_z']:.6f}",
                    f"- Short free-run QRC RMSE, original scale:   "
                    f"{result['short_free_run_rmse']:.6f}",
                ]
            )

    lines.extend(
        [
            "",
            "Takeaway:",
            "- Mackey-Glass is predicted very accurately by this small QRC.",
            "- NARMA10 is harder for the same reservoir settings.",
            "- This is the lesson: reservoir performance depends on the task,",
            "  even when the reservoir architecture is unchanged.",
            "",
            "Plot saved to:",
            str(plot_path),
            "",
            "Notes:",
            "- The reservoir is a quantum spin system.",
            "- The trained model is only the final linear readout.",
            "- The quantum density matrix rho is the model memory.",
            "- rho is never reset inside the time loop.",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    total_points_needed = TRAIN_STEPS + TEST_STEPS + 1

    mackey_glass = make_mackey_glass(total_points_needed)
    narma_input, narma_output = make_narma(total_points_needed)

    results = [
        run_qrc_task(
            task_name="Mackey-Glass",
            input_signal=mackey_glass,
            target_signal=mackey_glass,
            include_short_free_run=True,
        ),
        run_qrc_task(
            task_name=f"NARMA{NARMA_ORDER}",
            input_signal=narma_input,
            target_signal=narma_output,
            include_short_free_run=False,
        ),
    ]

    repo_dir = Path(__file__).resolve().parent
    results_dir = repo_dir / "results"
    results_dir.mkdir(exist_ok=True)
    plot_path = results_dir / "qrc_time_series_comparison.png"
    metrics_path = results_dir / "metrics.txt"

    make_comparison_plot(results, plot_path)
    summary = make_summary(results, plot_path.relative_to(repo_dir))

    print(summary)
    metrics_path.write_text(summary)


if __name__ == "__main__":
    main()
