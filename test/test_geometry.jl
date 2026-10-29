using Test: Test
using Random: Random
using CoarseGrainingEnergyFluxes: Geometry
using FlowTransformBindings: FlowTransformBindings as FTB

Test.@testset "Geometry Module Tests" begin

    Test.@testset "1. Exact Analytic Cases" begin
        square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

        # r = 0
        Test.@test Geometry.disk_fraction(square, 0.0) == 0.0
        Test.@test Geometry.disk_fraction(2.0, 2.0, 0.0) == 0.0
        Test.@test Geometry.disk_fraction(-1.0, 1.0, -1.0, 1.0, 0.0) == 0.0

        # Disk completely inside square (area = 4.0, disk = π * 0.5^2)
        expected_inscribed = π * 0.5^2 / 4.0
        Test.@test Geometry.disk_fraction(square, 0.5) ≈ expected_inscribed atol=1e-15
        Test.@test Geometry.disk_fraction(2.0, 2.0, 0.5) ≈ expected_inscribed atol=1e-15
        Test.@test Geometry.disk_fraction(-1.0, 1.0, -1.0, 1.0, 0.5) ≈ expected_inscribed atol=1e-15

        # Square completely inside disk (r > √2)
        Test.@test Geometry.disk_fraction(square, 1.5) == 1.0
        Test.@test Geometry.disk_fraction(2.0, 2.0, 1.5) == 1.0
        Test.@test Geometry.disk_fraction(-1.0, 1.0, -1.0, 1.0, 1.5) == 1.0

        # First-quadrant square [0, 1] × [0, 1] with r = 1.0 -> π/4
        q_sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        Test.@test Geometry.disk_fraction(q_sq, 1.0) ≈ π / 4.0 atol=1e-15
        Test.@test Geometry.disk_fraction(0.0, 1.0, 0.0, 1.0, 1.0) ≈ π / 4.0 atol=1e-15

        # Corner-miss test: box [0.8, 1.8]^2 overlaps [-1, 1]^2 AABB, but closest point (0.8, 0.8) has r^2 = 1.28 > 1
        far_sq = [(0.8, 0.8), (1.8, 0.8), (1.8, 1.8), (0.8, 1.8)]
        Test.@test Geometry.disk_fraction(far_sq, 1.0) == 0.0
        Test.@test Geometry.disk_fraction(0.8, 1.8, 0.8, 1.8, 1.0) == 0.0
    end

    # Both forms sum terms as large as the disk's area and the corners' products, so their rounding is a few
    # `eps` of those over the box's area.
    Test.@testset "2. Analytic Box vs. Polygon Agreement (1,000 Random Rectangles)" begin
        rng = Random.MersenneTwister(42)
        rounding(r, x, y, area) = 64 * eps() * (r^2 + x * y) / area
        for _ in 1:1000
            Lx, Ly = 0.1 + 4.0 * rand(rng), 0.1 + 4.0 * rand(rng)
            r = 3.0 * rand(rng)
            poly_c = [(-Lx/2, -Ly/2), (Lx/2, -Ly/2), (Lx/2, Ly/2), (-Lx/2, Ly/2)]
            Test.@test Geometry.disk_fraction(Lx, Ly, r) ≈ Geometry.disk_fraction(poly_c, r) atol = rounding(r, Lx, Ly, Lx * Ly)

            x1 = -2.5 + 5.0 * rand(rng)
            x2 = x1 + 0.05 + 2.5 * rand(rng)
            y1 = -2.5 + 5.0 * rand(rng)
            y2 = y1 + 0.05 + 2.5 * rand(rng)
            poly_b = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            Test.@test Geometry.disk_fraction(x1, x2, y1, y2, r) ≈ Geometry.disk_fraction(poly_b, r) atol =
                rounding(r, max(abs(x1), abs(x2)), max(abs(y1), abs(y2)), (x2 - x1) * (y2 - y1))
        end
    end

    # A simple quadrilateral is the union of the triangles on a diagonal with their orientations, whether or not
    # that diagonal lies inside it, so the overlaps add as signed areas. Vertices sorted by angle about a point
    # cross when one angular gap passes π; a crossed quadrilateral has no area fraction, so those are skipped.
    Test.@testset "3. Polygon Additivity / Triangulation Invariance" begin
        rng = Random.MersenneTwister(99)
        cross2(t) = (t[2][1] - t[1][1]) * (t[3][2] - t[1][2]) - (t[3][1] - t[1][1]) * (t[2][2] - t[1][2])
        side(a, b, c) = sign((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]))
        crosses(p, q, r, s) = side(p, q, r) != side(p, q, s) && side(r, s, p) != side(r, s, q)
        for _ in 1:500
            angles = sort!(rand(rng, 4) .* (2π))
            cx, cy = -1.0 + 2.0 * rand(rng), -1.0 + 2.0 * rand(rng)
            poly = [(cx + (0.4 + rand(rng)) * cos(θ), cy + (0.4 + rand(rng)) * sin(θ)) for θ in angles]
            r = 0.1 + 2.5 * rand(rng)
            (crosses(poly[1], poly[2], poly[3], poly[4]) || crosses(poly[2], poly[3], poly[4], poly[1])) && continue

            # Split quad into two triangles T1 = (1,2,3) and T2 = (1,3,4)
            t1 = [poly[1], poly[2], poly[3]]
            t2 = [poly[1], poly[3], poly[4]]
            s1, s2 = cross2(t1), cross2(t2)

            Test.@test Geometry.disk_fraction(poly, r) * (s1 + s2) ≈
                       Geometry.disk_fraction(t1, r) * s1 + Geometry.disk_fraction(t2, r) * s2 atol = 1e-13 * (π * r^2 + abs(s1) + abs(s2))
        end
    end

    nodes = (gl = FTB.gauss_legendre(Float64, 24); (gl.μ, gl.w))

    # A latitude–longitude grid over the whole sphere tiles every cap, so the cells' overlaps add to the cap's
    # area `2π(1 − cos θ)` on the unit sphere: caps smaller than a cell, wider than a hemisphere, centred on a pole
    # and across the grid's seam.
    Test.@testset "4. Cap ∩ latitude–longitude cells tile the cap" begin
        nλ, nφ = 36, 18
        λf = range(0.0, 2π; length = nλ + 1)
        φf = range(-π / 2, π / 2; length = nφ + 1)
        for (β, θ, λc) in ((0.3, 0.05, 0.1), (0.3, 0.7, 1.0), (1.5, 0.4, 2.0), (-1.2, 2.0, 5.9), (0.9, 2.9, 3.3),
                           (π / 2, 0.6, 0.0), (-0.4, 0.2, 0.0))
            total = 0.0
            for j in 1:nφ, i in 1:nλ
                a, b = λf[i] - λc, λf[i + 1] - λc
                total += (b - a) * (sin(φf[j + 1]) - sin(φf[j])) *
                         Geometry.cap_cell_fraction(a, b, φf[j], φf[j + 1], β, θ, nodes)
            end
            Test.@test total ≈ 2π * (1 - cos(θ)) rtol = 1e-10
        end
        Test.@test Geometry.cap_cell_fraction(-0.01, 0.01, 0.29, 0.31, 0.3, 0.5, nodes) === 1.0
        Test.@test Geometry.cap_cell_fraction(2.0, 2.1, 0.29, 0.31, 0.3, 0.5, nodes) === 0.0
    end

    # Boxes tiling space tile every ball, so their overlaps add to `4πR³/3`.
    Test.@testset "5. Ball ∩ boxes of a lattice tile the ball" begin
        h = (0.31, 0.27, 0.37)
        s = (0.11, -0.07, 0.05)
        R = 1.0
        n = ntuple(d -> ceil(Int, (R + h[d]) / h[d]) + 1, 3)
        total = 0.0
        for k in -n[3]:n[3], j in -n[2]:n[2], i in -n[1]:n[1]
            lo = (s[1] + (i - 0.5) * h[1], s[2] + (j - 0.5) * h[2], s[3] + (k - 0.5) * h[3])
            total += prod(h) * Geometry.ball_box_fraction(lo[1], lo[1] + h[1], lo[2], lo[2] + h[2], lo[3], lo[3] + h[3], R, nodes)
        end
        Test.@test total ≈ 4π / 3 * R^3 rtol = 1e-10
        Test.@test Geometry.ball_box_fraction(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, R, nodes) === 1.0
        Test.@test Geometry.ball_box_fraction(0.8, 0.9, 0.7, 0.9, 0.1, 0.2, R, nodes) === 0.0
    end

    # The gnomonic cubed sphere's cells are quadrilaterals with great-circle edges tiling the sphere, so their
    # overlaps with a cap add to its area. The cells are turned so the cap sits nowhere special; each cell's area
    # is the spherical excess of its two triangles (Van Oosterom & Strackee 1983).
    Test.@testset "6. Cap ∩ cubed-sphere cells tile the cap" begin
        n = 6
        α = range(-π / 4, π / 4; length = n + 1)
        k = (1.0, 2.0, 3.0) ./ sqrt(14.0)
        t = 0.7
        cr(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
        dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
        turn(v) = v .* cos(t) .+ cr(k, v) .* sin(t) .+ k .* (dot3(k, v) * (1 - cos(t)))
        unit(v) = v ./ sqrt(dot3(v, v))
        excess(a, b, c) = 2 * atan(abs(dot3(a, cr(b, c))), 1 + dot3(a, b) + dot3(b, c) + dot3(c, a))
        axes3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        faces = [(s .* axes3[d], axes3[mod1(d + 1, 3)], axes3[mod1(d + 2, 3)]) for d in 1:3 for s in (1.0, -1.0)]
        for θ in (0.05, 0.8, 2.2, 3.0)
            total = 0.0
            for (e0, e1, e2) in faces, j in 1:n, i in 1:n
                p(a, b) = turn(unit(e0 .+ tan(a) .* e1 .+ tan(b) .* e2))
                V = (p(α[i], α[j]), p(α[i + 1], α[j]), p(α[i + 1], α[j + 1]), p(α[i], α[j + 1]))
                total += (excess(V[1], V[2], V[3]) + excess(V[1], V[3], V[4])) * Geometry.cap_polygon_fraction(V, θ)
            end
            Test.@test total ≈ 2π * (1 - cos(θ)) rtol = 1e-9
        end
    end

    # A circle of latitude `φ` lies in the cap about `(0, β)` where `cos θ ≤ sin φ sin β + cos φ cos β cos λ`, so
    # arcs tiling it add to `2w`, `w` the longitude where equality holds.
    Test.@testset "7. Arcs of a circle of latitude tile its part inside a cap" begin
        nλ = 48
        λf = range(0.0, 2π; length = nλ + 1)
        for (φ, β, θ, λc) in ((0.3, 0.5, 0.4, 0.2), (-0.2, 0.1, 1.0, 3.1), (1.2, 1.0, 0.25, 6.0), (1.3, 1.0, 1.0, 1.0))
            w = acos(clamp((cos(θ) - sin(φ) * sin(β)) / (cos(φ) * cos(β)), -1.0, 1.0))
            total = sum(i -> (λf[i + 1] - λf[i]) * Geometry.cap_arc_fraction(λf[i] - λc, λf[i + 1] - λc, φ, β, θ), 1:nλ)
            Test.@test total ≈ 2w rtol = 1e-12
        end
    end

    Test.@testset "8. Segments tiling a line tile its part inside an interval" begin
        h, s, r = 0.37, 0.05, 1.3
        Test.@test sum(i -> h * Geometry.interval_fraction(s + (i - 0.5) * h, s + (i + 0.5) * h, r), -6:6) ≈ 2r rtol = 1e-14
    end

    # Sectors of latitude, longitude and radius tile space, so their overlaps with a ball add to `4πR³/3`: balls
    # near a pole, containing the origin, and thin against the radius.
    Test.@testset "9. Ball ∩ spherical shell sectors tile the ball" begin
        for (β, r0, R, nλ, nφ, nr) in ((0.3, 1.0, 0.2, 24, 12, 10), (1.4, 1.0, 0.5, 24, 12, 10), (-0.7, 1.0, 1.4, 18, 9, 12),
                                       (0.0, 6.371, 0.05, 72, 36, 6))
            λf = range(0.0, 2π; length = nλ + 1)
            φf = range(-π / 2, π / 2; length = nφ + 1)
            rf = range(max(0.0, r0 - R - 0.07), r0 + R + 0.05; length = nr + 1)
            total = sum(Iterators.product(1:nλ, 1:nφ, 1:nr)) do (i, j, k)
                (λf[i + 1] - λf[i]) * (sin(φf[j + 1]) - sin(φf[j])) * (rf[k + 1]^3 - rf[k]^3) / 3 *
                Geometry.shell_cell_fraction(λf[i] - 0.4, λf[i + 1] - 0.4, φf[j], φf[j + 1], rf[k], rf[k + 1], β, r0, R, nodes)
            end
            Test.@test total ≈ 4π / 3 * R^3 rtol = 1e-10
        end
    end
end