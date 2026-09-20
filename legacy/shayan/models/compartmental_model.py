"""
Compartmental cable equation solver for neuronal morphologies.

Based on Borst & Meier (2019) Current Biology - "Extreme Compartmentalization
in a Drosophila Amacrine Cell"

Implements passive cable theory using the Hines matrix method with sparse
linear algebra for efficient solving.

Physical Model:
    - Passive membrane (no voltage-gated channels)
    - Linear cable equation: C·dV/dt = -M·V + I_inj
    - Implicit Euler time integration for stability

Units:
    - Spatial: micrometers (μm)
    - Resistance: Ohm·cm (Ra), Ohm·cm² (Rm)
    - Capacitance: μF/cm² (Cm)
    - Current: Amperes (A)
    - Voltage: millivolts (mV)
    - Time: seconds (s)
"""

import numpy as np
import navis
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict
import warnings

if TYPE_CHECKING:
    from ..morphology.compartmentalization import Compartmentalization


class CompartmentalModel:
    """
    Passive cable equation solver for compartmentalized neurons.

    This class builds a conductance matrix from neuron geometry and solves
    the cable equation for voltage distribution given current injection.

    Parameters:
        comp: Compartmentalization object containing neuron geometry
        Rm: Specific membrane resistance (Ω·cm²). Default 8000.0 from Borst 2019
        Ra: Specific axial resistance (Ω·cm). Default 400.0 from Borst 2019
        Cm: Specific membrane capacitance (μF/cm²). Default 0.6 from Borst 2019

    Attributes:
        comp: The compartmentalization object
        Rm, Ra, Cm: Biophysical parameters
        n_comps: Number of compartments
        M: Sparse conductance matrix (N×N)
        memcap: Membrane capacitance vector (N,)

    Example:
        >>> from shayan.morphology.loader import load_skeleton
        >>> from shayan.morphology.segmentation import segment_natural
        >>> from shayan.models.compartmental_model import CompartmentalModel
        >>>
        >>> # Load and segment neuron
        >>> neuron = load_skeleton('data/...', [neuron_id])[0]
        >>> comp = segment_natural(neuron)
        >>>
        >>> # Build cable model
        >>> model = CompartmentalModel(comp)
        >>>
        >>> # Inject current into compartment 0, solve for steady-state voltage
        >>> V = model.solve_steady_state(
        ...     injection_comps=[0],
        ...     currents=[10e-12]  # 10 pA
        ... )
        >>> print(f"Voltage at soma: {V[0]:.2f} mV")

    Notes:
        - All spatial measurements from compartmentalization are assumed to be in μm
        - Current injections should be in Amperes (A)
        - Output voltages are in millivolts (mV)
        - The conductance matrix M is built once at initialization and cached
    """

    def __init__(
        self,
        comp: 'Compartmentalization',
        Rm: float = 8000.0,  # Ω·cm²
        Ra: float = 400.0,   # Ω·cm
        Cm: float = 0.6      # μF/cm²
    ):
        """Initialize compartmental model with geometry and biophysical parameters."""
        self.comp = comp
        self.Rm = Rm
        self.Ra = Ra
        self.Cm = Cm

        # Extract geometry and build matrices
        self.n_comps = len(comp.compartments)
        self._diameters, self._lengths, self._parent_indices = self._extract_geometry()

        n_zero = int((self._lengths == 0.0).sum())
        if n_zero > 0:
            warnings.warn(
                f"{n_zero} compartment(s) have zero length. They receive no axial "
                "coupling, which makes the conductance matrix singular and the "
                "solvers return NaN. Merge or drop zero-length compartments before "
                "building the model.",
                stacklevel=2,
            )

        self.M, self.memcap = self._build_conductance_matrix()

    def _extract_geometry(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract geometric properties from compartmentalization.

        Returns:
            diameters: (N,) array of compartment diameters in μm
            lengths: (N,) array of compartment lengths in μm
            parent_indices: (N,) array of parent compartment indices (-1 for root)
        """
        # Get compartments dataframe
        compartments = self.comp.compartments

        # Extract diameters 
        diameters = compartments['radius_mean'].to_numpy() * 2.0

        # Extract lengths 
        lengths = compartments['length'].to_numpy()

        # Extract parent ids
        parent_indices = compartments['parent_id'].to_numpy()

        return diameters, lengths, parent_indices


    def _build_conductance_matrix(self) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build sparse conductance matrix M and membrane capacitance vector.

        Returns:
            M: Sparse conductance matrix (N×N) in CSR format
            memcap: Membrane capacitance vector (N,)
        """
        M = sparse.lil_matrix((self.n_comps, self.n_comps))

        # Loop through and construct off diagonal elements - axial conductances
        for i in range(1, self.n_comps):
            # Skip zero length segments or those with no parents
            if self._lengths[i] == 0.0 or self._parent_indices[i] == -1:
                continue
            # Axial conductance
            mean_diam = (self._diameters[i] + self._diameters[self._parent_indices[i]]) / 2.0
            cross_area = (mean_diam**2.0) * np.pi / (4.0)
            M[i, self._parent_indices[i]] = -10**(-4) * cross_area / (self.Ra * self._lengths[i]) 
            M[self._parent_indices[i], i] = M[i, self._parent_indices[i]]
        
        # Loop through and calculate on diagonal elements - leak conductances 
        for i in range(self.n_comps):
            g_leak = (np.pi * self._diameters[i] * self._lengths[i]) / (self.Rm * 10**8)
            M[i, i] = g_leak - np.sum(M[i])

        # Calculate membrane capacitance 
        memcap = self._diameters * np.pi * self._lengths * self.Cm * (10**-6)/(10**8)

        return sparse.csr_matrix(M), memcap


    def solve_steady_state(
        self,
        injection_comps: List[int],
        currents: List[float]
    ) -> np.ndarray:
        """
        Solve for steady-state voltage distribution.

        Solves: M·V = I_inj

        Args:
            injection_comps: List of compartment IDs where current is injected
            currents: List of current amplitudes in Amperes (A)

        Returns:
            V: Voltage at each compartment in millivolts (mV), shape (N,)

        Example:
            >>> # Inject 10 pA into compartment 0
            >>> V = model.solve_steady_state(
            ...     injection_comps=[0],
            ...     currents=[10e-12]
            ... )
            >>>
            >>> # Inject into multiple compartments
            >>> V = model.solve_steady_state(
            ...     injection_comps=[0, 5, 10],
            ...     currents=[10e-12, 5e-12, 8e-12]
            ... )
        """
        # Create current vector
        J = np.zeros(self.n_comps)
        J[injection_comps] = currents

        # Solve optimization problem 
        V = spsolve(self.M, J)

        return V * 1000 # convert to mV

    def solve_time_dependent(
        self,
        injection_comps: List[int],
        current_timeseries: np.ndarray,
        dt: float = 0.001,
        t_max: float = 0.1
    ) -> np.ndarray:
        """
        Solve time-dependent cable equation with implicit Euler.

        Solves: (M + C/Δt)·V[t] = (C/Δt)·V[t-1] + I_inj[t]

        Args:
            injection_comps: List of compartment IDs where current is injected
            current_timeseries: Current injection over time, shape (len(injection_comps), n_timesteps)
                              or (n_timesteps,) if single compartment. Units: Amperes (A)
            dt: Timestep in seconds. Default 0.001 (1 ms)
            t_max: Total simulation time in seconds. Default 0.1 (100 ms)

        Returns:
            V: Voltage at each compartment over time, shape (n_comps, n_timesteps)
               Units: millivolts (mV)

        Example:
            >>> # Constant current injection for 50 ms
            >>> n_timesteps = int(0.050 / 0.001)  # 50 timesteps
            >>> current = np.ones(n_timesteps) * 10e-12  # 10 pA constant
            >>> V = model.solve_time_dependent(
            ...     injection_comps=[0],
            ...     current_timeseries=current,
            ...     dt=0.001,
            ...     t_max=0.050
            ... )
            >>>
            >>> # Time-varying current (e.g., pulse)
            >>> current = np.zeros(n_timesteps)
            >>> current[10:30] = 10e-12  # 20 ms pulse
            >>> V = model.solve_time_dependent([0], current)
        """
        n_t = int(t_max / dt)
        if current_timeseries.ndim == 1:
            timesteps = current_timeseries.shape[0]
        else:
            timesteps = current_timeseries.shape[1]
        if n_t != timesteps:
            raise ValueError("Incompatible time discretizations.")

        # Building time dependent conductance matrix
        M_t = self.M.copy()
        M_t.setdiag(M_t.diagonal() + self.memcap/dt)

        # Voltage traces
        V = np.zeros((self.n_comps, n_t))

        # Injected currents
        J = np.zeros((self.n_comps, n_t))
        J[injection_comps] = current_timeseries

        # Solve cable equation at each time step 
        for t in range(1, n_t):
            target = V[:, t-1]*self.memcap/dt + J[:, t]
            V[:, t] = spsolve(M_t, target)

        return V * 1000 # convert to mV

    def get_input_resistance(self, compartment_id: int) -> float:
        """
        Calculate input resistance at a compartment.

        Input resistance is V/I when 1 Ampere is injected into the compartment.

        Args:
            compartment_id: Compartment ID to measure input resistance

        Returns:
            R_in: Input resistance in GigaOhms (GΩ)

        Example:
            >>> R_in = model.get_input_resistance(0)
            >>> print(f"Input resistance at soma: {R_in:.2f} GΩ")
        """
        V = self.solve_steady_state([compartment_id], currents=[1])
        R_in = V[compartment_id] / (10**12) # Convert to GigaOhms and adjust for mV output
        return R_in

    def plot_voltage_distribution(
        self,
        V: np.ndarray,
        title: str = "Voltage Distribution",
        **kwargs
    ):
        """
        Visualize voltage distribution on neuron morphology.

        Args:
            V: Voltage at each compartment (mV), shape (n_comps,)
            title: Plot title
            **kwargs: Additional arguments passed to visualization function

        Returns:
            Plotly figure showing neuron colored by voltage
        """
        import plotly.express as px

        # Normalize voltage values to [0, 1] for colormap
        V_norm = (V - V.min()) / (V.max() - V.min()) if V.max() > V.min() else np.zeros_like(V)

        # Create color mapping using plasma colormap
        color_scale = px.colors.sequential.Plasma
        colors = {}
        for i, val in enumerate(V_norm):
            idx = int(val * (len(color_scale) - 1))
            colors[i] = color_scale[idx]

        # Plot skeleton segments colored by voltage
        node_colors = {}
        for comp_id, node_ids in self.comp.node_mapping.items():
            for node_id in node_ids:
                node_colors[node_id] = colors[comp_id]

        # Use navis to plot with custom colors
        fig_navis = navis.plot3d(
            self.comp.neuron,
            backend='plotly',
            inline=False,
            color=node_colors,
            **kwargs
        )

        # Update title and add voltage info
        fig_navis.update_layout(
            title=f'{title}<br>Voltage range: {V.min():.2f} - {V.max():.2f} mV',
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Z (μm)',
                aspectmode='data'
            ),
            hovermode='closest'
        )

        return fig_navis

    def analyze_compartmentalization(self, injection_comp: int = 0, current: float = 10e-12) -> Dict[str, float]:
        """
        Analyze electrical isolation between compartments.

        Injects current into one compartment and measures voltage attenuation
        to neighbors, similar to Borst 2019 Figure 3A.

        Args:
            injection_comp: Compartment to inject current into
            current: Current amplitude to inject (A). Default 10 pA.

        Returns:
            Dictionary with analysis results:
                - 'input_resistance': Input resistance at injection site (GΩ)
                - 'voltage_at_injection': Voltage at injection site (mV)
                - 'mean_neighbor_voltage': Mean voltage at neighboring compartments (mV)
                - 'attenuation_percent': Mean voltage drop to neighbors (%)

        Example:
            >>> results = model.analyze_compartmentalization(injection_comp=0)
            >>> print(f"Voltage drops to {results['attenuation_percent']:.1f}% in neighbors")
        """
        # Inject current and solve
        V = self.solve_steady_state([injection_comp], [current])

        # Get voltage at injection site
        V_inj = V[injection_comp]

        # Find neighboring compartments (children and parent)
        neighbors = []

        # Find parent
        if self._parent_indices[injection_comp] != -1:
            neighbors.append(self._parent_indices[injection_comp])

        # Find children (compartments where injection_comp is the parent)
        for i in range(self.n_comps):
            if self._parent_indices[i] == injection_comp:
                neighbors.append(i)

        # Calculate mean voltage at neighbors
        if len(neighbors) > 0:
            mean_neighbor_V = np.mean([V[n] for n in neighbors])
            attenuation_percent = (mean_neighbor_V / V_inj) * 100.0
        else:
            mean_neighbor_V = 0.0
            attenuation_percent = 0.0

        # Get input resistance
        R_in = self.get_input_resistance(injection_comp)

        return {
            'input_resistance': R_in,
            'voltage_at_injection': float(V_inj),
            'mean_neighbor_voltage': float(mean_neighbor_V),
            'attenuation_percent': float(attenuation_percent),
            'num_neighbors': len(neighbors)
        }


def scan_parameter_space(
    comp: 'Compartmentalization',
    Ra_values: np.ndarray,
    Rm_values: np.ndarray,
    injection_comp: int = 0,
    current: float = 10e-12
) -> np.ndarray:
    """
    Scan Ra and Rm parameter space (like Borst 2019 Figure 3B).

    For each (Ra, Rm) pair:
        1. Build model
        2. Inject current
        3. Measure mean voltage in neighboring compartments
        4. Calculate attenuation ratio

    Args:
        comp: Compartmentalization object
        Ra_values: Array of Ra values to test (Ω·cm)
        Rm_values: Array of Rm values to test (Ω·cm²)
        injection_comp: Which compartment to inject into
        current: Current amplitude (A)

    Returns:
        attenuation_matrix: (len(Rm_values), len(Ra_values)) array
                          Values are mean voltage in neighbors as % of injection site

    Example:
        >>> Ra_vals = np.linspace(100, 500, 10)
        >>> Rm_vals = np.linspace(1000, 10000, 10)
        >>> attenuation = scan_parameter_space(comp, Ra_vals, Rm_vals)
        >>>
        >>> # Plot heatmap
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(attenuation, origin='lower', vmin=0, vmax=70)
        >>> plt.colorbar(label='Mean Vm [% of max]')
        >>> plt.xlabel('Ra (Ω·cm)')
        >>> plt.ylabel('Rm (kΩ·cm²)')
    """
    # Initialize results matrix
    attenuation_matrix = np.zeros((len(Rm_values), len(Ra_values)))

    # Scan parameter space
    for j, Rm in enumerate(Rm_values):
        for i, Ra in enumerate(Ra_values):
            # Build model with current parameters
            model = CompartmentalModel(comp, Rm=Rm, Ra=Ra)

            # Analyze compartmentalization
            results = model.analyze_compartmentalization(injection_comp, current)

            # Store attenuation percentage
            attenuation_matrix[j, i] = results['attenuation_percent']

    return attenuation_matrix
