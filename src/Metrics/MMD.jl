using KernelFunctions, LinearAlgebra, Distances
using MLUtils

export mmd
"""
    mmd(X, Y; kernel=SqExponentialKernel()) -> Float64

Compute squared Maximum Mean Discrepancy (MMD²) between datasets `X` and `Y`
using a positive-definite kernel from `KernelFunctions.jl`.

Arguments
---------
- `X`, `Y`  :: AbstractMatrix – rows = features, columns = samples.
- `kernel` :: KernelFunctions.Kernel – e.g. `SqExponentialKernel()`

Returns
-------
- `Float64` – biased estimate of MMD²
"""
function mmd(X::AbstractMatrix, Y::AbstractMatrix;
    kernel::Kernel=SqExponentialKernel())::Float64
    n, m = numobs(X), numobs(Y)

    Kxx = sum(kernelmatrix(kernel, X, obsdim=2)) / n^2
    Kyy = sum(kernelmatrix(kernel, Y, obsdim=2)) / m^2
    Kxy = sum(kernelmatrix(kernel, X, Y, obsdim=2)) / (n * m)

    return Kxx + Kyy - 2 * Kxy
end
