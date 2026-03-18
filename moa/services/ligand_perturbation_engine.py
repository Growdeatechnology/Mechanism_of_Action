# ============================================================
# ligand_perturbation_engine.py
# INDUSTRY-GRADE MOA COUNTERFACTUAL ENGINE
# SMARTS + BRICS + SCAFFOLD HOPPING + FILTERS + DIVERSITY
# ============================================================

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdChemReactions
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
import itertools


# ============================================================
# GLOBAL SETTINGS
# ============================================================

FP_RADIUS = 2
FP_BITS = 2048
DIVERSITY_THRESHOLD = 0.85
MAX_BRICS_CANDIDATES = 30


# ============================================================
# Utility — Safe Sanitize
# ============================================================

def _sanitize_and_smiles(mol):
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


# ============================================================
# Chemical Realism Filters (Medicinal Chemistry Guardrails)
# ============================================================

def _passes_filters(new_mol, ref_mol):

    try:
        if abs(Descriptors.MolWt(new_mol) - Descriptors.MolWt(ref_mol)) > 150:
            return False

        if abs(new_mol.GetNumHeavyAtoms() - ref_mol.GetNumHeavyAtoms()) > 12:
            return False

        if abs(new_mol.GetRingInfo().NumRings() - ref_mol.GetRingInfo().NumRings()) > 3:
            return False

        if Descriptors.NumHDonors(new_mol) > 8:
            return False

        if Descriptors.NumHAcceptors(new_mol) > 12:
            return False

        if Descriptors.MolLogP(new_mol) > 7:
            return False

        return True

    except:
        return False


# ============================================================
# Structural Delta Label (UI Meaningful)
# ============================================================

def _structural_delta_label(ref_mol, new_mol):

    ref_fp = AllChem.GetMorganFingerprint(ref_mol, 2)
    new_fp = AllChem.GetMorganFingerprint(new_mol, 2)

    ref_bits = set(ref_fp.GetNonzeroElements().keys())
    new_bits = set(new_fp.GetNonzeroElements().keys())

    added = len(new_bits - ref_bits)
    removed = len(ref_bits - new_bits)

    if added > removed:
        return "Added structural features"
    elif removed > added:
        return "Removed structural features"
    else:
        return "Local structural modification"


# ============================================================
# SMARTS Reaction Library (Production Safe)
# ============================================================

SMARTS_LIBRARY = [

    # Carbonyl family
    ("Carbonyl → Thiocarbonyl", "[C:1]=O>>[C:1]=S"),
    ("Amide → Ketone", "[C:1](=O)N>>[C:1](=O)C"),
    ("Ketone → Amide", "[C:1](=O)C>>[C:1](=O)N"),

    # Nitrile conversions
    ("Nitrile → Amide", "[C:1]#N>>[C:1](=O)N"),

    # Halogen swaps
    ("Cl → F", "[Cl:1]>>[F:1]"),
    ("F → Cl", "[F:1]>>[Cl:1]"),
    ("Cl → Br", "[Cl:1]>>[Br:1]"),

    # Oxygen / Sulfur swaps
    ("O → S", "[O:1]>>[S:1]"),
    ("S → O", "[S:1]>>[O:1]"),
]


def smarts_transforms(ref_mol):

    variants = []

    for label, smarts in SMARTS_LIBRARY:

        try:
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            products = rxn.RunReactants((ref_mol,))

            for prod_tuple in products:

                prod = prod_tuple[0]
                smi = _sanitize_and_smiles(prod)

                if not smi:
                    continue

                new_mol = Chem.MolFromSmiles(smi)

                if new_mol and _passes_filters(new_mol, ref_mol):
                    variants.append((label, smi))

        except:
            continue

    return variants


# ============================================================
# BRICS Recombination (Controlled)
# ============================================================

def brics_recombine(ref_mol):

    variants = []

    fragments = list(BRICS.BRICSDecompose(ref_mol))
    frag_mols = [Chem.MolFromSmiles(f) for f in fragments if Chem.MolFromSmiles(f)]

    if len(frag_mols) < 2:
        return []

    combos = list(itertools.combinations(frag_mols, 2))

    for combo in combos[:MAX_BRICS_CANDIDATES]:

        try:
            builder = BRICS.BRICSBuild(combo)

            for m in builder:

                smi = _sanitize_and_smiles(m)
                if not smi:
                    continue

                new_mol = Chem.MolFromSmiles(smi)

                if new_mol and _passes_filters(new_mol, ref_mol):
                    variants.append(("Fragment recombination", smi))

        except:
            continue

    return variants


# ============================================================
# Scaffold Hopping (Preserve Side Chains)
# ============================================================

SCAFFOLD_LIBRARY = [
    Chem.MolFromSmiles("c1ccncc1"),   # pyridine
    Chem.MolFromSmiles("c1ncccn1"),   # pyrimidine
    Chem.MolFromSmiles("c1ccoc1"),    # furan
    Chem.MolFromSmiles("c1ccsc1"),    # thiophene
]


def scaffold_hopping(ref_mol):

    variants = []

    scaffold = MurckoScaffold.GetScaffoldForMol(ref_mol)

    if not scaffold:
        return []

    for new_scaffold in SCAFFOLD_LIBRARY:

        try:
            replaced = Chem.ReplaceSubstructs(
                ref_mol,
                scaffold,
                new_scaffold,
                replaceAll=False
            )

            for m in replaced:

                smi = _sanitize_and_smiles(m)
                if not smi:
                    continue

                new_mol = Chem.MolFromSmiles(smi)

                if new_mol and _passes_filters(new_mol, ref_mol):
                    variants.append(("Scaffold modification", smi))

        except:
            continue

    return variants


# ============================================================
# Diversity Filter
# ============================================================

def _diversity_filter(variants):

    kept = []
    fps = []

    for item in variants:

        mol = Chem.MolFromSmiles(item["SMILES"])
        if not mol:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, FP_BITS)

        keep = True

        for existing_fp in fps:
            if DataStructs.TanimotoSimilarity(fp, existing_fp) > DIVERSITY_THRESHOLD:
                keep = False
                break

        if keep:
            kept.append(item)
            fps.append(fp)

    return kept


# ============================================================
# MASTER FUNCTION
# ============================================================

def generate_perturbations(smiles, max_variants=40):

    ref_mol = Chem.MolFromSmiles(smiles)

    if not ref_mol:
        return []

    all_variants = []

    # Layer 1 — SMARTS Transforms
    all_variants += smarts_transforms(ref_mol)

    # Layer 2 — BRICS
    all_variants += brics_recombine(ref_mol)

    # Layer 3 — Scaffold Hopping
    all_variants += scaffold_hopping(ref_mol)

    # ========================================================
    # Deduplicate + UI Meaningful Label
    # ========================================================

    seen = set()
    unique = []

    for label, smi in all_variants:

        if smi in seen or smi == smiles:
            continue

        seen.add(smi)

        new_mol = Chem.MolFromSmiles(smi)

        if not new_mol:
            continue

        unique.append({
            "Modification_Type": label,  # real chemical transform
            "Delta_Label": _structural_delta_label(ref_mol, new_mol),
            "Original_SMILES": smiles,
            "SMILES": smi
        })

    # ========================================================
    # Diversity Control
    # ========================================================

    unique = _diversity_filter(unique)

    return unique[:max_variants]