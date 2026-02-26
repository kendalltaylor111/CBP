Constraint-Based Physicalism

------------------

How to Read This Paper (CBP_v13.pdf):

This paper does not attempt to derive subjective experience from neural activity, solve the hard problem in the reductive sense, or identify a neural correlate of consciousness. It does something different: it asks what kind of physical fact consciousness would have to be if it is a physical fact at all, and finds that the answer dissolves the problem that motivated the question.

The argument begins with an observation about physics. Organisms tracking rough environmental signals (power-law spectra with α < 2) face a metabolic wall: discrete, snapshot-based architectures require orders of magnitude more energy than continuous, phase-locked architectures to achieve the same fidelity. At biologically relevant fidelities, the discrete path exceeds the brain's energy budget—for the roughest natural signals, it fails at any power level. Evolution was forced into a specific dynamical regime: a constraint-maintained temporal parallax phase that actively bridges the delay between environmental flow and internal representation. Numerical simulation confirms the metabolic wall with discrete-to-continuous power ratios exceeding 150× at biological fidelities.

The philosophical move is then to notice what this phase is. Its parameters systematically determine the major features of phenomenal experience: coherence persistence determines unity, proximity to critical delay determines temporal texture, bifurcation collapse determines the transition to unconsciousness, constitutive irreversibility determines the arrow of subjective time, and continuous informational geometry determines qualitative richness. Once the phase is fully specified, no phenomenal fact remains undetermined. The paper argues that this forces an identity: the phase does not produce consciousness—it is consciousness, in the same sense that temperature is mean molecular kinetic energy.

This identity predicts the hard problem rather than being threatened by it. If experience is the continuous informational geometry of the phase, and third-person description is lossy with respect to that geometry, then third-person accounts will necessarily seem to leave experience out. The explanatory gap is a compression artifact—a gap in description, not in ontology. The zombie thought experiment fails not because zombies are physically impossible, but because the specification is incoherent: it demands the outputs of the parallax phase while subtracting the phase itself.

The paper develops this argument with formal precision, including a resolution of how consciousness can involve a sharp existence threshold (a phase transition) while phenomenology is graded (varying elaboration above threshold), a response to Kripke's anti-physicalist argument via the compression artifact, and a demonstration that the resulting ontology is more parsimonious than dualism, panpsychism, emergentism, functionalism, or eliminativism. Falsifiable predictions for neurophysiology and AI architecture are provided.

------------------

*The Python script cbp_simulation_standalone.py generates the simulation data reported in Section 11 of the Constraint‑Based Physicalism (CBP) paper. Run with Python 3.8+ and the required dependency numpy. If you wish to use plotting features (--plot or --grid), you also need matplotlib.
