# Theory

## Coarse-Graining Framework

The coarse-grained (filtered) velocity field at scale ℓ is defined as:

```
ū_ℓ(x) = ∫ G_ℓ(x, x') u(x') dA(x')
```

where G_ℓ is a normalized filter kernel with characteristic scale ℓ.

### Sub-Scale Stress Tensor

The sub-scale stress captures the effect of scales smaller than ℓ:

```
τ_ℓ = (u ⊗ u)̄_ℓ − ū_ℓ ⊗ ū_ℓ
```

This is analogous to the Reynolds stress in RANS, but defined locally at every point and every scale.

### Cross-Scale Energy Flux

The filtered kinetic energy equation (Aluie 2011) gives:

```
∂ₜ(½|ū_ℓ|²) + ... = −Π_ℓ + transport terms
```

where:

```
Π_ℓ(x) = −τ_ℓ : S̄_ℓ = −Σᵢⱼ τᵢⱼ S̄ᵢⱼ
```

and S̄ = ½(∇ū + (∇ū)ᵀ) is the large-scale strain rate tensor.

- **Π > 0**: Forward cascade (energy transferred from large → small scales)
- **Π < 0**: Inverse cascade (energy transferred from small → large scales)
- **⟨Π⟩ > 0**: Net forward cascade across the domain at scale ℓ

## Filter Kernels

### Top-Hat Kernel (Default)

```
G_ℓ(r) = 1/(πℓ²)  if r ≤ ℓ
          0          otherwise
```

Sharp cutoff at scale ℓ. Most common in the literature (Aluie et al. 2018, Storer et al. 2022).

### Gaussian Kernel

```
G_ℓ(r) = (1/(2πσ²)) exp(−r²/(2σ²)),   σ = ℓ/2
```

Smooth, infinitely differentiable. Better spectral localization but broader real-space support.

### Sharp Spectral Kernel

```
Ĝ_ℓ(k) = 1  if |k| ≤ 2π/ℓ
           0  otherwise
```

Ideal low-pass filter in Fourier space. Requires FFT extension. Perfect scale separation but Gibbs ringing in physical space.

## Spherical Geometry

On the sphere S² with radius R, the convolution uses great-circle distance:

```
d(x, x') = R · arccos(sin φ sin φ' + cos φ cos φ' cos(λ−λ'))
```

The area element is dA = R² cos φ dλ dφ.

### Commutativity on the Sphere

Aluie (2019) proves that filtering vector fields by converting to Cartesian components and filtering each as a scalar does **NOT** commute with ∇ on S²:

```
G * (∇·u) ≠ ∇·(G * u)    on S²
```

**Implications for this package:**
- For **non-divergent** velocity (∇·u = 0, e.g., geostrophic), the planetary Cartesian approach is exact (Storer et al. 2022)
- For **general** velocity with divergent components, the correct approach requires Helmholtz decomposition — see [HelmholtzDecomposition.jl](https://github.com/jbphyswx/HelmholtzDecomposition.jl)

## References

- Aluie, H. (2011). Compressible turbulence: the cascade and its locality. *Physical Review Letters*, 106(17). doi:10.1016/j.physd.2011.06.001
- Aluie, H. (2019). Convolutions on the sphere: commutation with differential operators. *GEM*, 10(1). doi:10.1007/s13137-019-0123-9
- Aluie, H., Hecht, M., & Vallis, G. K. (2018). Mapping the energy cascade in the North Atlantic Ocean. *Journal of Physical Oceanography*, 48(8). doi:10.1175/JPO-D-17-0100.1
- Storer, B. A. et al. (2022). Global energy spectrum of the general oceanic circulation. *Nature Communications*, 13, 5314. doi:10.1038/s41467-022-33031-3
- Buzzicotti, M. et al. (2023). Spatio-temporal coarse-graining decomposition. *Science Advances*, 9(45). doi:10.1126/sciadv.adi7420
