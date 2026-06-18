using CSV, DataFrames, Tables, Dates, MLUtils, CodecZlib

export covid_moonshot_dataloader, merck_dataloader, merck_train_dataloader, merck_split_dataloader, load_outer_cv_subset

function covid_moonshot_dataloader(
    filepath::String,
    target_col::Symbol
)::DataFrame
    df = CSV.read(filepath, DataFrame)

    # 2. Keep only rows with valid SMILES and ASSAYED == true
    mask_smiles = .!ismissing.(df.SMILES)
    df_clean = df[mask_smiles, :]

    required_cols = [:SMILES, :ORDER_DATE, :SHIPMENT_DATE, target_col]
    missing_cols = setdiff(required_cols, propertynames(df_clean))
    if !isempty(missing_cols)
        error("Missing required columns: ", join(missing_cols, ", "))
    end

    latest_date(d1::Union{Missing,Date}, d2::Union{Missing,Date}) =
        d1 === missing ? d2 :
        d2 === missing ? d1 :
        max(d1, d2)

    df_clean[!, :date] = latest_date.(df_clean.ORDER_DATE, df_clean.SHIPMENT_DATE)

    selected = df_clean[:, [:SMILES, :date, target_col]]
    dropmissing!(selected)

    return selected
end

const MERCK_TARGET_COL = :Act
const MERCK_EXCLUDE_COLS = [:MOLECULE, MERCK_TARGET_COL]

# Resolve a CSV path: prefers <path>.gz if present, otherwise <path>.
function _resolve_csv_path(path::String)
    isfile(path * ".gz") && return path * ".gz"
    isfile(path) && return path
    throw(ArgumentError("file not found: $path (also tried .gz)"))
end

# Private helper: read one Merck CSV (plain or gzip-compressed) into a features×samples
# Int32 matrix. Uses Tables.columntable as the CSV sink, bypassing DataFrame entirely.
function _merck_load_csv(path::String)
    resolved = _resolve_csv_path(path)
    stream   = endswith(resolved, ".gz") ? GzipDecompressorStream(open(resolved)) : open(resolved)
    ct = try
        CSV.read(stream, Tables.columntable; typemap=Dict(Int64 => Int32))
    finally
        close(stream)
    end

    y            = Vector{Float64}(ct[MERCK_TARGET_COL])
    feature_cols = [c for c in Tables.columnnames(ct) if c ∉ MERCK_EXCLUDE_COLS]
    n_features   = length(feature_cols)
    n_samples    = length(y)

    X = Matrix{Int32}(undef, n_features, n_samples)
    for (i, col) in enumerate(feature_cols)
        X[i, :] = ct[col]
    end

    return X, y, feature_cols   # feature_cols is Vector{Symbol}
end

"""
    merck_dataloader(base_folder::String, dataset_name::String) -> (train=(X, y), test=(X, y))

Loads the training and test datasets for a given `dataset_name` from the `base_folder`.
It expects two CSV files named as `"<dataset_name>_training_disguised.csv"` and
`"<dataset_name>_test_disguised.csv"`.

Ensures both train and test feature matrices have the same columns. If test is missing
any train columns, they are added with zero values.

Features are stored as `Int32` (Merck descriptors are small integers; Int32 halves memory
vs the Int64 CSV default). Labels are `Float64`.

# Returns
`(train=(X=Matrix{Int32}, y=Vector{Float64}), test=(X=Matrix{Int32}, y=Vector{Float64}))`
where `X` is laid out as (features × samples).
"""
function _merck_load_ips_csv(path::String, feature_cols::Vector{Symbol})
    resolved = _resolve_csv_path(path)
    stream   = endswith(resolved, ".gz") ? GzipDecompressorStream(open(resolved)) : open(resolved)
    ct = try
        CSV.read(stream, Tables.columntable; typemap=Dict(Int64 => Int32))
    finally
        close(stream)
    end
    y         = Vector{Float64}(ct[MERCK_TARGET_COL])
    n_samples = length(y)
    X         = Matrix{Int32}(undef, length(feature_cols), n_samples)
    for (i, col) in enumerate(feature_cols)
        if hasproperty(ct, col)
            X[i, :] = ct[col]
        else
            X[i, :] .= zero(Int32)
        end
    end
    X, y
end

function merck_dataloader(base_folder::String, dataset_name::String)
    train_path = joinpath(base_folder, "$(dataset_name)_training_disguised.csv")
    test_path  = joinpath(base_folder, "$(dataset_name)_test_disguised.csv")

    X_train, y_train, feature_cols = _merck_load_csv(train_path)
    X_test, y_test = _merck_load_ips_csv(test_path, feature_cols)
    GC.gc()

    (train=(X=X_train, y=y_train), test=(X=X_test, y=y_test))
end

"""
    merck_train_dataloader(base_folder::String, dataset_name::String) -> (X, y)

Like `merck_dataloader` but loads only the training CSV — use this when the IPS/test
matrix is not needed (e.g., splitting calculations) to avoid loading several GB of
data that would otherwise be immediately discarded.
"""
function merck_train_dataloader(base_folder::String, dataset_name::String)
    train_path = joinpath(base_folder, "$(dataset_name)_training_disguised.csv")
    X, y, _ = _merck_load_csv(train_path)  # _resolve_csv_path handles .gz fallback + missing error
    return (X=X, y=y)
end

"""
    merck_split_dataloader(base_folder, dataset_name, train_idx, test_idx)
        -> (train=(X,y), test=(X,y), ips=(X,y))

Memory-efficient loader for AD annotation. Reads the training CSV once into a
columntable, builds only the `train_idx` and `test_idx` submatrices (avoiding
the full N×features matrix), frees the columntable, then loads the full IPS set.

Use instead of `merck_dataloader` when only a small subset of training samples
is needed (e.g. sub-sampled experiments).
"""
function merck_split_dataloader(
    base_folder::String,
    dataset_name::String,
    train_idx::AbstractVector{<:Integer},
    test_idx::AbstractVector{<:Integer},
)
    train_path = joinpath(base_folder, "$(dataset_name)_training_disguised.csv")
    test_path  = joinpath(base_folder, "$(dataset_name)_test_disguised.csv")

    resolved = _resolve_csv_path(train_path)
    stream   = endswith(resolved, ".gz") ? GzipDecompressorStream(open(resolved)) : open(resolved)
    ct = try
        CSV.read(stream, Tables.columntable; typemap=Dict(Int64 => Int32))
    finally
        close(stream)
    end

    y_all        = ct[MERCK_TARGET_COL]
    feature_cols = [c for c in Tables.columnnames(ct) if c ∉ MERCK_EXCLUDE_COLS]
    n_features   = length(feature_cols)

    X_train = Matrix{Int32}(undef, n_features, length(train_idx))
    X_test  = Matrix{Int32}(undef, n_features, length(test_idx))
    for (i, col) in enumerate(feature_cols)
        v = ct[col]
        X_train[i, :] = v[train_idx]
        X_test[i, :]  = v[test_idx]
    end
    y_train = Vector{Float64}(y_all[train_idx])
    y_test  = Vector{Float64}(y_all[test_idx])
    ct = nothing; GC.gc()

    X_ips, y_ips = _merck_load_ips_csv(test_path, feature_cols)
    GC.gc()

    (
        train = (X=X_train, y=y_train),
        test  = (X=X_test,  y=y_test),
        ips   = (X=X_ips,   y=y_ips),
    )
end

"""
    load_outer_cv_subset(data_root, split_root, dataset, rep, fold;
                         subsampled_cv_path=nothing) -> (X, y, idx)

Load the Merck training set for `dataset` and return the subset of
observations belonging to the outer-train partition of `(rep, fold)`.

When `subsampled_cv_path` is provided it must be a JLD2 file written by
`scripts/subsample_cv.jl` containing a pre-drawn `"idx"` vector of global
indices.  Those indices are used directly instead of the full outer-train
partition, so every splitter that runs on the same `(rep, fold)` operates
on exactly the same compounds.

When `subsampled_cv_path` is `nothing` the full outer-train partition is
used (original behaviour).

Only the training CSV is loaded (not the IPS/test set), so this is
significantly cheaper than calling `merck_dataloader` when the test set
is not needed.

# Returns
- `X`: feature matrix (features × subset samples, Int32)
- `y`: label vector for subset samples
- `idx`: global indices into the full training set corresponding to X/y
"""
function load_outer_cv_subset(
    data_root::AbstractString,
    split_root::AbstractString,
    dataset::AbstractString,
    rep::Integer,
    fold::Integer;
    subsampled_cv_path::Union{Nothing,AbstractString}=nothing,
)
    data = merck_train_dataloader(data_root, dataset)

    cv_idx = if subsampled_cv_path !== nothing
        jldopen(subsampled_cv_path, "r") do f
            Vector{Int}(f["idx"])
        end
    else
        load_cv(split_root, dataset, rep, fold).train
    end

    Xsub = getobs(data.X, cv_idx)
    ysub = getobs(data.y, cv_idx)

    return (X=Xsub, y=ysub, idx=cv_idx)
end
