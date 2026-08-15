"""
BB84 Quantum Key Distribution protocol simulation.

Connects the FSOC channel simulation to a BB84 protocol using Qiskit.
The atmospheric channel effects (loss, phase noise, background) are mapped
to quantum noise models applied to single-qubit prepare-and-measure circuits.

Noise mapping:
    - Channel loss (1 - eta)  →  amplitude damping  (photon lost → random outcome)
    - Phase variance          →  depolarizing error  (polarization scrambling)
    - Background / dark counts →  bit-flip error      (false detections)

Key rate formula (Shor-Preskill bound):
    SKR = max(0, eta/2 * [1 - h(QBER) - f_EC * h(QBER)])
    where h(x) = -x*log2(x) - (1-x)*log2(1-x) is the binary entropy.
"""

from dataclasses import dataclass, field
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

from channel_metrics import FSOCChannelMetrics


@dataclass
class BB84Result:
    num_sent: int
    raw_key_length: int
    sifted_key_length: int
    qber: float
    secret_key_rate: float       # bits per channel use
    secure: bool                 # True if QBER < 11%
    alice_sifted_key: np.ndarray = field(repr=False)
    bob_sifted_key: np.ndarray = field(repr=False)


def _binary_entropy(x):
    """Binary entropy function h(x) = -x*log2(x) - (1-x)*log2(1-x)."""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def _build_noise_model(channel: FSOCChannelMetrics):
    """Build a Qiskit NoiseModel from FSOC channel metrics.

    The noise is applied to the measurement gate to model:
    1. Loss: with probability (1 - eta), the photon is lost and the detector
       outputs a random bit (modeled as 50% bit-flip on the lost fraction).
    2. Phase noise: depolarizing channel from residual wavefront phase variance.
    3. Background: additional symmetric bit-flip from dark counts / stray light.
    """
    # Combined error probability
    p_loss_flip = (1.0 - channel.eta) * 0.5  # lost photons → random outcome
    p_depol = channel.depolarizing_prob
    p_background = min(channel.background_counts, 0.5)  # cap at 50%

    # Total effective bit-flip probability (union of independent sources)
    # P(error) = 1 - (1-p1)(1-p2)(1-p3)
    p_no_error = (1.0 - p_loss_flip) * (1.0 - p_depol) * (1.0 - p_background)
    p_total_error = 1.0 - p_no_error
    p_total_error = max(0.0, min(p_total_error, 1.0))

    # Build noise model: depolarizing error on measurement
    noise_model = NoiseModel()
    if p_total_error > 0:
        error = depolarizing_error(p_total_error, 1)
        noise_model.add_all_qubit_quantum_error(error, ['measure'])

    return noise_model, p_total_error


class BB84Simulator:
    """Simulate BB84 QKD protocol over an FSOC channel.

    Parameters
    ----------
    channel : FSOCChannelMetrics
        Physical channel metrics from the FSOC simulation.
    num_qubits : int
        Number of qubits (photon pulses) Alice sends.
    error_correction_efficiency : float
        Error correction inefficiency factor f_EC (typically ~1.16).
    """

    def __init__(
        self,
        channel: FSOCChannelMetrics,
        num_qubits: int = 10000,
        error_correction_efficiency: float = 1.16,
    ):
        self.channel = channel
        self.num_qubits = num_qubits
        self.f_ec = error_correction_efficiency
        self._backend = AerSimulator()

    def run(self) -> BB84Result:
        """Execute the full BB84 protocol.

        Steps:
            1. Alice generates random bits and random bases (Z or X).
            2. Each qubit is prepared, sent through noisy channel, and measured by Bob.
            3. Bases are compared (sifting).
            4. QBER is estimated from a test subset.
            5. Secret key rate is computed via Shor-Preskill bound.
        """
        n = self.num_qubits

        # Step 1: Alice's random choices
        alice_bits = np.random.randint(0, 2, n)
        alice_bases = np.random.randint(0, 2, n)  # 0=Z, 1=X

        # Step 2: Bob's random measurement bases
        bob_bases = np.random.randint(0, 2, n)

        # Build noise model
        noise_model, _ = _build_noise_model(self.channel)

        # Step 3: Simulate each qubit through the noisy channel
        bob_results = np.zeros(n, dtype=int)

        # Batch qubits into circuits for efficiency
        batch_size = 100
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            circuits = []

            for i in range(batch_start, batch_end):
                qc = QuantumCircuit(1, 1)

                # Alice prepares
                if alice_bits[i] == 1:
                    qc.x(0)
                if alice_bases[i] == 1:  # X basis
                    qc.h(0)

                # Bob measures in his chosen basis
                if bob_bases[i] == 1:  # X basis
                    qc.h(0)
                qc.measure(0, 0)

                circuits.append(qc)

            # Run batch with noise
            result = self._backend.run(
                circuits, noise_model=noise_model, shots=1
            ).result()

            for idx, i in enumerate(range(batch_start, batch_end)):
                counts = result.get_counts(idx)
                bob_results[i] = int(list(counts.keys())[0])

        # Step 4: Sifting — keep only matching bases
        matching = alice_bases == bob_bases
        alice_sifted = alice_bits[matching]
        bob_sifted = bob_results[matching]
        sifted_len = len(alice_sifted)

        # Step 5: QBER estimation
        if sifted_len > 0:
            errors = np.sum(alice_sifted != bob_sifted)
            qber = float(errors / sifted_len)
        else:
            qber = 0.5

        # Step 6: Secret key rate (Shor-Preskill bound)
        # SKR = max(0, eta/2 * [1 - h(QBER) - f_EC * h(QBER)])
        h_qber = _binary_entropy(qber)
        skr_per_pulse = max(0.0, self.channel.eta / 2.0 * (1.0 - h_qber - self.f_ec * h_qber))

        secure = qber < 0.11  # BB84 security threshold

        return BB84Result(
            num_sent=n,
            raw_key_length=n,
            sifted_key_length=sifted_len,
            qber=qber,
            secret_key_rate=skr_per_pulse,
            secure=secure,
            alice_sifted_key=alice_sifted,
            bob_sifted_key=bob_sifted,
        )

    def summary(self, result: BB84Result) -> str:
        """Return a human-readable summary of the BB84 result."""
        lines = [
            "=" * 50,
            "BB84 QKD Simulation Results",
            "=" * 50,
            f"Channel transmittance (eta): {self.channel.eta:.4f}",
            f"Channel Strehl ratio:        {self.channel.strehl_ratio:.4f}",
            f"Phase variance:              {self.channel.phase_variance:.4f} rad^2",
            f"Depolarizing probability:    {self.channel.depolarizing_prob:.6f}",
            "-" * 50,
            f"Qubits sent:                 {result.num_sent}",
            f"Sifted key length:           {result.sifted_key_length}",
            f"QBER:                        {result.qber:.4f} ({result.qber*100:.2f}%)",
            f"Secret key rate (per pulse): {result.secret_key_rate:.6f}",
            f"Secure (QBER < 11%):         {'YES' if result.secure else 'NO'}",
            "=" * 50,
        ]
        return "\n".join(lines)
