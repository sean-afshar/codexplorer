"""Passive compartmental cable model (Borst & Meier 2019 style), ported from shayan.

Units: compartment geometry in micrometres; Rm ohm cm^2; Ra ohm cm; Cm uF/cm^2;
injected current in amperes; voltages returned in millivolts; time in seconds.
The conductance matrix is built once, vectorized, and factorized once with
``splu`` so repeated solves (parameter scans, time steps) are cheap.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from connexplorer.morph.segment import Compartments


def _inject_vector(n: int, inject) -> np.ndarray:
    J = np.zeros(n)
    if isinstance(inject, dict):
        for k, v in inject.items():
            J[int(k)] += float(v)
    else:
        comps, currents = inject
        J[np.asarray(comps, dtype=int)] += np.asarray(currents, dtype=float)
    return J


class Cable:
    def __init__(self, comp: Compartments, Rm: float = 8000.0, Ra: float = 400.0, Cm: float = 0.6):
        self.comp = comp
        self.Rm, self.Ra, self.Cm = float(Rm), float(Ra), float(Cm)
        self.n = len(comp)
        self._lengths = comp.lengths
        self._diameters = comp.diameters
        if np.any(self._lengths <= 0):
            raise ValueError("compartments with zero length; use segment(..., min_length_um > 0)")
        self.M, self.memcap = self._build()

    def _build(self) -> tuple[sp.csr_matrix, np.ndarray]:
        axial = self.comp.hines(self.Ra)  # -g off-diagonal, +sum g on diagonal
        g_leak = np.pi * self._diameters * self._lengths / (self.Rm * 1e8)
        M = (axial + sp.diags(g_leak)).tocsc()
        memcap = np.pi * self._diameters * self._lengths * self.Cm * 1e-6 / 1e8
        return M, memcap

    @cached_property
    def _lu(self):
        return splu(self.M.tocsc())

    def steady_state(self, inject) -> np.ndarray:
        """Solve M V = I. ``inject`` is ``{compartment: amperes}`` or ``(compartments, currents)``. Returns mV."""
        return self._lu.solve(_inject_vector(self.n, inject)) * 1e3

    def transient(self, inject: dict[int, np.ndarray], dt: float = 1e-3, v0: np.ndarray | None = None) -> np.ndarray:
        """Implicit Euler: (M + C/dt) V_t = (C/dt) V_{t-1} + I_t. ``inject`` maps compartment -> current trace (A).

        Returns (n_compartments, n_steps) in mV.
        """
        traces = {int(k): np.asarray(v, dtype=float) for k, v in inject.items()}
        n_t = max(len(v) for v in traces.values())
        if any(len(v) != n_t for v in traces.values()):
            raise ValueError("all current traces must have the same length")
        J = np.zeros((self.n, n_t))
        for k, v in traces.items():
            J[k] += v
        c_dt = self.memcap / dt
        lu = splu((self.M + sp.diags(c_dt)).tocsc())
        V = np.zeros((self.n, n_t))
        if v0 is not None:
            V[:, 0] = np.asarray(v0) / 1e3
        for t in range(1, n_t):
            V[:, t] = lu.solve(V[:, t - 1] * c_dt + J[:, t])
        return V * 1e3

    def input_resistance(self, compartment: int = 0) -> float:
        """Input resistance in gigaohms."""
        V = self.steady_state({compartment: 1.0})
        return float(V[compartment] / 1e3 / 1e9)

    def attenuation(self, compartment: int = 0, current: float = 10e-12) -> dict:
        """Voltage at the injection site and at its neighbours (Borst 2019 Fig. 3A)."""
        V = self.steady_state({compartment: current})
        parents = self.comp.parents
        neighbours = list(self.comp.children(compartment))
        if parents[compartment] >= 0:
            neighbours.append(int(parents[compartment]))
        v_inj = float(V[compartment])
        mean_nb = float(np.mean(V[neighbours])) if neighbours else 0.0
        return {
            "input_resistance_GOhm": self.input_resistance(compartment),
            "voltage_at_injection_mV": v_inj,
            "mean_neighbour_voltage_mV": mean_nb,
            "attenuation_percent": 100.0 * mean_nb / v_inj if v_inj else 0.0,
            "n_neighbours": len(neighbours),
        }

    def __repr__(self) -> str:
        return f"Cable({self.n} compartments, Rm={self.Rm:g} ohm cm2, Ra={self.Ra:g} ohm cm, Cm={self.Cm:g} uF/cm2)"


def scan(comp: Compartments, Ra, Rm, compartment: int = 0, current: float = 10e-12, Cm: float = 0.6) -> np.ndarray:
    """Attenuation (% of injection-site voltage at neighbours) over a (len(Rm), len(Ra)) grid."""
    Ra, Rm = np.atleast_1d(Ra), np.atleast_1d(Rm)
    out = np.zeros((len(Rm), len(Ra)))
    for j, rm in enumerate(Rm):
        for i, ra in enumerate(Ra):
            out[j, i] = Cable(comp, Rm=rm, Ra=ra, Cm=Cm).attenuation(compartment, current)["attenuation_percent"]
    return out
