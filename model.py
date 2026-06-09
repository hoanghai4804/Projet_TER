# model.py
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def build_model():
    """
    Construit le réseau bayésien pour la détection
    d'attaques GPS Spoofing sur UAV.

    DAG: S → {C, D, A, P}
    S = Spoofing (variable cachée)
    C = Cohérence GPS/IMU
    D = Anomalie Doppler
    A = Anomalie Altitude
    P = Saut de Pseudorange
    """

    # ── Structure du DAG ──────────────────────
    model = DiscreteBayesianNetwork([
        ("S", "C"),
        ("S", "D"),
        ("S", "A"),
        ("S", "P")
    ])

    # ── CPT de S ──────────────────────────────
    cpd_S = TabularCPD(
        variable="S",
        variable_card=2,
        values=[[0.95],
                [0.05]],
        state_names={"S": [0, 1]}
    )

    # ── CPT de C ──────────────────────────────
    cpd_C = TabularCPD(
        variable="C",
        variable_card=2,
        values=[[0.95, 0.10],
                [0.05, 0.90]],
        evidence=["S"],
        evidence_card=[2],
        state_names={"C": [0, 1], "S": [0, 1]}
    )

    # ── CPT de D ──────────────────────────────
    cpd_D = TabularCPD(
        variable="D",
        variable_card=2,
        values=[[0.97, 0.30],
                [0.03, 0.70]],
        evidence=["S"],
        evidence_card=[2],
        state_names={"D": [0, 1], "S": [0, 1]}
    )

    # ── CPT de A ──────────────────────────────
    cpd_A = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.98, 0.60],
                [0.02, 0.40]],
        evidence=["S"],
        evidence_card=[2],
        state_names={"A": [0, 1], "S": [0, 1]}
    )

    # ── CPT de P ──────────────────────────────
    cpd_P = TabularCPD(
        variable="P",
        variable_card=2,
        values=[[0.96, 0.35],
                [0.04, 0.65]],
        evidence=["S"],
        evidence_card=[2],
        state_names={"P": [0, 1], "S": [0, 1]}
    )

    # ── Assembler ─────────────────────────────
    model.add_cpds(cpd_S, cpd_C, cpd_D, cpd_A, cpd_P)
    assert model.check_model(), "Modèle invalide !"
    print("Modele valide")

    # ── Inference engine ──────────────────────
    inference = VariableElimination(model)

    return model, inference


if __name__ == "__main__":

    model, inference = build_model()

    print("\n=== Vérification des scénarios ===\n")

    result = inference.query(
        variables=["S"],
        evidence={"C": 1},
        show_progress=False
    )
    print(f"P(S=1 | C=1)              = {result.values[1]:.3f}")

    result = inference.query(
        variables=["S"],
        evidence={"C": 1, "D": 1},
        show_progress=False
    )
    print(f"P(S=1 | C=1, D=1)         = {result.values[1]:.3f}")

    result = inference.query(
        variables=["S"],
        evidence={"C": 1, "D": 1, "A": 1, "P": 1},
        show_progress=False
    )
    print(f"P(S=1 | C=1,D=1,A=1,P=1) = {result.values[1]:.3f}")