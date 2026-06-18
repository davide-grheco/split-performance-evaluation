using MoleculeFlow: mol_from_smiles, morgan_fingerprint

export rdkit_fp

"""
    rdkit_fp(smiles; radius=2, nbits=2048) -> Union{Vector{Bool}, Missing}

Compute a Morgan (circular) fingerprint for a single SMILES string using
MoleculeFlow/RDKit.

# Arguments
- `smiles`: SMILES string of the molecule.
- `radius`: Morgan radius (number of bond hops). Default 2 (≈ ECFP4).
- `nbits`: Length of the bit vector. Default 2048.

Returns `missing` if the SMILES cannot be parsed.
"""
function rdkit_fp(smiles::String; radius::Int=2, nbits::Int=2048)::Union{Vector{Bool},Missing}
    mol = mol_from_smiles(smiles)
    return morgan_fingerprint(mol; radius=radius, nbits=nbits)
end

"""
    rdkit_fp(smiles; radius=2, nbits=2048) -> Vector{Union{Vector{Bool}, Missing}}

Batch variant of `rdkit_fp`: compute Morgan fingerprints for a vector of
SMILES strings. Elements that cannot be parsed are returned as `missing`.
"""
function rdkit_fp(smiles::Vector{String}; radius::Int=2, nbits::Int=2048)::Vector{Union{Vector{Bool},Missing}}
    mols = mol_from_smiles.(smiles)
    return morgan_fingerprint(mols; radius=radius, nbits=nbits)
end
