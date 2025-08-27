# main.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Pauli
from qiskit_aer import AerSimulator
from qiskit.qasm3 import dumps
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.library.standard_gates import HGate, XGate, YGate, ZGate, SGate, TGate, RXGate, RYGate, RZGate, CXGate, CZGate, SwapGate
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import io

def get_bloch_vectors(qc):
    try:
        # Validate circuit
        if qc.num_qubits == 0:
            raise ValueError("Circuit has no qubits.")
        for instruction in qc.data:
            if instruction.operation.name == 'measure':
                raise ValueError("Circuit contains measurements, which are not supported for statevector simulation.")
            if instruction.operation.name not in ['u', 'cx', 'h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'cz', 'swap']:
                st.warning(f"Unsupported instruction '{instruction.operation.name}' detected. Attempting decomposition.")

        # Decompose circuit to basis gates [u, cx]
        basis_gates = ['u', 'cx']
        pm = PassManager(BasisTranslator(StandardEquivalenceLibrary, basis_gates))
        qc_decomposed = pm.run(qc)
        
        # Log decomposed circuit instructions and QASM
        st.write("Decomposed circuit instructions:")
        instruction_list = [f"{instr.operation.name} on qubits {[q._index for q in instr.qubits]}" for instr in qc_decomposed.data]
        st.write(instruction_list)
        st.write("Decomposed circuit QASM:")
        st.code(dumps(qc_decomposed), language='text')
        
        # Run simulation with explicit statevector saving
        sim = AerSimulator(method='statevector')
        qc_decomposed.save_statevector()
        result = sim.run(qc_decomposed, shots=1).result()
        
        # Debug simulation result
        st.write("Simulation result data:", result.data(0))
        
        if 'statevector' not in result.data(0):
            raise ValueError(f"No statevector available in simulation result. Decomposed circuit: {instruction_list}")
        state = result.get_statevector(qc_decomposed)
        dm = DensityMatrix(state)
        n = qc_decomposed.num_qubits
        bloch_list = []
        st.write("Full density matrix:", dm.data)  # Added for debugging
        for i in range(n):
            rho = partial_trace(dm, list(set(range(n)) - {i}))
            st.write(f"Reduced density matrix for Qubit {i}: {rho.data}")  # Debug
            x = rho.expectation_value(Pauli('X')).real
            y = rho.expectation_value(Pauli('Y')).real
            z = rho.expectation_value(Pauli('Z')).real
            # Normalize and handle numerical precision
            length = np.sqrt(x**2 + y**2 + z**2)
            if length > 1 + 1e-10:  # Allow small numerical error
                x, y, z = x / length, y / length, z / length
            elif length < 1e-10:  # Treat as zero within numerical precision
                x, y, z = 0, 0, 0
            bloch_list.append([x, y, z])
            st.write(f"Bloch vector for Qubit {i}: [{x}, {y}, {z}]")  # Debug
        return bloch_list, state
    except Exception as e:
        raise ValueError(f"Simulation error: {str(e)}")

def plot_bloch_3d_qubits(bloch_vectors_list):
    n = len(bloch_vectors_list)
    fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'scene'} for _ in range(n)]],
                        subplot_titles=[f'Qubit {i}' for i in range(n)])
    for i, vec in enumerate(bloch_vectors_list):
        # Sphere surface
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2 * np.pi, 40)
        theta, phi = np.meshgrid(theta, phi)
        x_s = np.sin(theta) * np.cos(phi)
        y_s = np.sin(theta) * np.sin(phi)
        z_s = np.cos(theta)
        fig.add_trace(go.Surface(x=x_s, y=y_s, z=z_s, opacity=0.1, colorscale='Blues',
                                 showscale=False, lighting=dict(ambient=0.9)), row=1, col=i+1)
        
        # Axes lines
        axes = [[1,0,0,'x','red'], [0,1,0,'y','green'], [0,0,1,'z','blue']]
        for ax in axes:
            fig.add_trace(go.Scatter3d(x=[0, ax[0]], y=[0, ax[1]], z=[0, ax[2]],
                                       mode='lines', line=dict(color=ax[4], width=4)), row=1, col=i+1)
            fig.add_trace(go.Scatter3d(x=[ax[0]*1.05], y=[ax[1]*1.05], z=[ax[2]*1.05],
                                       mode='text', text=[ax[3]], textfont=dict(size=12)), row=1, col=i+1)
        
        # |0> and |1> labels
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1.1], mode='text', text=['|0⟩'],
                                   textfont=dict(size=16)), row=1, col=i+1)
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[-1.1], mode='text', text=['|1⟩'],
                                   textfont=dict(size=16)), row=1, col=i+1)
        
        # Bloch vector line
        x_v, y_v, z_v = vec
        fig.add_trace(go.Scatter3d(x=[0, x_v], y=[0, y_v], z=[0, z_v],
                                   mode='lines', line=dict(color='black', width=5)), row=1, col=i+1)
        
        # Arrow head
        fig.add_trace(go.Cone(x=[x_v], y=[y_v], z=[z_v], u=[x_v*0.1], v=[y_v*0.1], w=[z_v*0.1],
                              sizemode='absolute', sizeref=0.2, showscale=False,
                              colorscale=[[0, 'black'], [1, 'black']]), row=1, col=i+1)
        
        # Enhance visualization for mixed states
        length = np.sqrt(x_v**2 + y_v**2 + z_v**2)
        if length < 1e-10:  # Maximally mixed state (numerical zero)
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0],
                                       mode='markers', marker=dict(color='black', size=6, symbol='circle')),
                          row=1, col=i+1)
            st.write(f"Qubit {i} is in a maximally mixed state (Bloch vector at center).")
        elif length < 1:  # Partially mixed state
            scale_factor = max(0.05, 0.1 * length)  # Ensure visibility
            fig.add_trace(go.Cone(x=[x_v], y=[y_v], z=[z_v], u=[x_v*scale_factor/length], v=[y_v*scale_factor/length], w=[z_v*scale_factor/length],
                                  sizemode='absolute', sizeref=0.1, showscale=False,
                                  colorscale=[[0, 'black'], [1, 'black']]), row=1, col=i+1)
            st.write(f"Qubit {i} is in a mixed state (Bloch vector length {length:.4f} < 1).")
        
        # Update scene
        fig.update_scenes(xaxis_visible=True, yaxis_visible=True, zaxis_visible=True,
                          camera_eye=dict(x=1.6, y=1.6, z=0.6),
                          aspectmode='cube', row=1, col=i+1)
    
    fig.update_layout(height=600, width=400 * n, title="Bloch Spheres for Each Qubit")
    return fig

st.title("Quantum State Visualizer")

if 'qc' not in st.session_state:
    st.session_state.qc = None
if 'gates' not in st.session_state:
    st.session_state.gates = []

# Add test circuit button
if st.button("Test Simple Circuit"):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    st.session_state.qc = qc
    st.session_state.gates = []
    st.rerun()

option = st.selectbox("Choose option", [
    "Random Multi-qubit Circuit Generation",
    "Manual Multi-qubit Circuit",
    "Import/Manual Code"
])

if option == "Random Multi-qubit Circuit Generation":
    num_qubits = st.number_input("Number of qubits (1-10)", min_value=1, max_value=10, value=2)
    depth = st.number_input("Circuit depth", min_value=1, max_value=20, value=3)
    if st.button("Generate Random Circuit"):
        try:
            gate_map = {
                'h': HGate(), 'x': XGate(), 'y': YGate(), 'z': ZGate(),
                's': SGate(), 't': TGate(),
                'rx': lambda: RXGate(np.random.uniform(0, 2 * np.pi)),
                'ry': lambda: RYGate(np.random.uniform(0, 2 * np.pi)),
                'rz': lambda: RZGate(np.random.uniform(0, 2 * np.pi)),
                'cx': CXGate(), 'cz': CZGate(), 'swap': SwapGate()
            }
            qc = QuantumCircuit(num_qubits)
            applied_gates = []
            for _ in range(depth):
                for q in range(num_qubits):
                    gate_name = np.random.choice(list(gate_map.keys()))
                    gate = gate_map[gate_name]
                    if gate_name in ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz']:
                        gate_instance = gate() if callable(gate) else gate
                        qc.append(gate_instance, [q])
                        applied_gates.append(f"{gate_name} on qubit {q}")
                    elif gate_name in ['cx', 'cz', 'swap'] and num_qubits > 1:
                        target = np.random.choice([i for i in range(num_qubits) if i != q])
                        qc.append(gate, [q, target])
                        applied_gates.append(f"{gate_name} on qubits ({q}, {target})")
            st.session_state.qc = qc
            st.session_state.gates = []
            st.write("Applied gates:", applied_gates)
        except Exception as e:
            st.error(f"Error generating circuit: {e}")

elif option == "Manual Multi-qubit Circuit":
    num_qubits = st.number_input("Number of qubits (1-10)", min_value=1, max_value=10, value=2)
    st.write("Add gates one by one:")
    
    with st.form("Add Gate"):
        gate_type = st.selectbox("Gate Type", ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cz', 'swap'])
        params = []
        qubits = []
        if gate_type in ['rx', 'ry', 'rz']:
            angle = st.number_input("Angle (radians)", value=0.0)
            qubit = st.number_input("Qubit", min_value=0, max_value=num_qubits-1, value=0)
            params = [angle]
            qubits = [qubit]
        elif gate_type in ['cx', 'cz', 'swap']:
            control = st.number_input("Control Qubit", min_value=0, max_value=num_qubits-1, value=0)
            target = st.number_input("Target Qubit", min_value=0, max_value=num_qubits-1, value=1)
            if control == target:
                st.error("Control and target qubits must be different.")
                st.stop()
            qubits = [control, target]
        else:
            qubit = st.number_input("Qubit", min_value=0, max_value=num_qubits-1, value=0)
            qubits = [qubit]
        
        submit = st.form_submit_button("Add Gate")
        if submit:
            st.session_state.gates.append({'type': gate_type, 'params': params, 'qubits': qubits})
            st.rerun()
    
    if st.button("Clear All Gates"):
        st.session_state.gates = []
        st.rerun()
    
    # Build qc from gates
    if st.session_state.gates:
        try:
            qc = QuantumCircuit(num_qubits)
            for gate in st.session_state.gates:
                getattr(qc, gate['type'])(*gate['params'], *gate['qubits'])
            st.session_state.qc = qc
        except Exception as e:
            st.error(f"Error building circuit: {e}")

elif option == "Import/Manual Code":
    code = st.text_area("Paste Qiskit Python code defining 'qc' (e.g., qc = QuantumCircuit(2); qc.h(0); ...)", height=200)
    if st.button("Load Circuit"):
        try:
            local_vars = {}
            exec(code, {"QuantumCircuit": QuantumCircuit}, local_vars)
            if 'qc' in local_vars:
                st.session_state.qc = local_vars['qc']
                st.session_state.gates = []
            else:
                st.error("No 'qc' defined in the code.")
        except Exception as e:
            st.error(f"Error executing code: {e}")

# Display circuit if available
if st.session_state.qc is not None:
    qc = st.session_state.qc
    st.subheader("Quantum Circuit")
    try:
        # Attempt Matplotlib drawer with default style
        fig_circ = circuit_drawer(qc, output='mpl', style='default')
        buf = io.BytesIO()
        fig_circ.savefig(buf, format='png')
        plt.close(fig_circ)
        st.image(buf)
    except Exception as e:
        st.error(f"Error displaying circuit (e.g., missing pylatexenc): {e}. Please ensure 'pylatexenc' is installed via 'pip install pylatexenc' in requirements.txt and redeploy.")

    try:
        qasm_code = dumps(qc)
        st.subheader("QASM Code")
        st.code(qasm_code, language='text')
        st.download_button("Download QASM", qasm_code, file_name="circuit.qasm")
    except Exception as e:
        st.error(f"Error generating QASM: {e}")
    
    if st.button("Visualize on Bloch Spheres"):
        try:
            bloch_list, state = get_bloch_vectors(qc)
            fig_bloch = plot_bloch_3d_qubits(bloch_list)
            st.plotly_chart(fig_bloch, use_container_width=True)
            
            st.subheader("State Vector")
            st.write(state)
        except Exception as e:
            st.error(f"Error in visualization: {e}")