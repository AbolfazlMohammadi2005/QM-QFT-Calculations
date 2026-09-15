import sympy as sp

from einsteinpy.symbolic import (
    MetricTensor,
    ChristoffelSymbols
)


# ============================================================
# 1. Coordinates and symbols
# ============================================================

t = sp.symbols("t", real=True)

r = sp.symbols(
    "r",
    positive=True,
    real=True
)

theta = sp.symbols(
    "theta",
    real=True
)

phi = sp.symbols(
    "phi",
    real=True
)

coords = (
    t,
    r,
    theta,
    phi
)


# ============================================================
# 2. Physical parameters
# ============================================================

M = sp.Symbol(
    "M",
    positive=True,
    real=True
)

m = sp.Symbol(
    "m",
    real=True
)


# ============================================================
# 3. Metric function
#
# F(r) is kept symbolic.
#
# The program does NOT substitute a specific expression
# such as 1 - 2M/r into F(r).
# ============================================================

F = sp.Function("F")(r)


# ============================================================
# 4. Metric
#
# Signature: (+, -, -, -)
#
# ds^2 =
# F(r) dt^2
# - dr^2/F(r)
# - r^2 dtheta^2
# - r^2 sin^2(theta) dphi^2
# ============================================================

g = sp.Matrix([
    [
        F,
        0,
        0,
        0
    ],

    [
        0,
        -1 / F,
        0,
        0
    ],

    [
        0,
        0,
        -r**2,
        0
    ],

    [
        0,
        0,
        0,
        -r**2 * sp.sin(theta)**2
    ]
])


# ============================================================
# 5. Minkowski metric
# ============================================================

eta = sp.diag(
    1,
    -1,
    -1,
    -1
)


# ============================================================
# 6. Simplification
# ============================================================

def simp(expr):

    expr = sp.trigsimp(expr)
    expr = sp.cancel(expr)
    expr = sp.factor(expr)

    return expr


# ============================================================
# 7. LaTeX expression
# ============================================================

def latex_expr(expr):

    return sp.latex(
        simp(expr)
    )


# ============================================================
# 8. Matrix -> LaTeX rows
# ============================================================

def latex_matrix_rows(matrix):

    rows = []

    for i in range(matrix.rows):

        row = []

        for j in range(matrix.cols):

            value = simp(
                matrix[i, j]
            )

            row.append(
                sp.latex(value)
            )

        rows.append(row)

    return rows


# ============================================================
# 9. Determine whether a matrix needs a landscape page
#
# This is intentionally conservative.
#
# Large/long matrices are placed on a separate landscape page.
# Normal matrices stay normal-size in portrait mode.
# ============================================================

def matrix_needs_landscape(matrix):

    rows = latex_matrix_rows(matrix)

    total_length = 0
    maximum_length = 0

    for row in rows:

        row_length = sum(
            len(item)
            for item in row
        )

        total_length += row_length

        maximum_length = max(
            maximum_length,
            row_length
        )

    # Long expressions inside a matrix
    if maximum_length > 75:
        return True

    # Large total matrix content
    if total_length > 230:
        return True

    return False


# ============================================================
# 10. Write matrix
#
# No resizebox is used.
#
# If the matrix is large:
#
#     new page
#     landscape
#     matrix
#     end landscape
#
# Otherwise it is printed normally.
# ============================================================

def write_matrix(
    tex,
    name,
    matrix
):

    zero_matrix = sp.zeros(
        matrix.rows,
        matrix.cols
    )

    # --------------------------------------------------------
    # Completely zero matrix
    # --------------------------------------------------------

    if matrix == zero_matrix:

        tex.write("\\[\n")
        tex.write("\\begin{aligned}\n")
        tex.write(
            f"{name} &= 0\n"
        )
        tex.write("\\end{aligned}\n")
        tex.write("\\]\n\n")

        return

    rows = latex_matrix_rows(
        matrix
    )

    landscape = matrix_needs_landscape(
        matrix
    )

    # --------------------------------------------------------
    # Landscape page
    # --------------------------------------------------------

    if landscape:

        tex.write("\\clearpage\n")
        tex.write("\\begin{landscape}\n")

        tex.write("\\section*{")
        tex.write(
            name.replace(
                "\\",
                ""
            )
        )
        tex.write("}\n\n")

        tex.write("\\[\n")
        tex.write("\\begin{pmatrix}\n")

        for i, row in enumerate(rows):

            line = " & ".join(row)

            if i < len(rows) - 1:

                tex.write(
                    line
                    + r" \\"
                    + "\n"
                )

            else:

                tex.write(
                    line
                    + "\n"
                )

        tex.write("\\end{pmatrix}\n")
        tex.write("\\]\n\n")

        tex.write("\\end{landscape}\n")
        tex.write("\\clearpage\n\n")

        return

    # --------------------------------------------------------
    # Normal portrait page
    # --------------------------------------------------------

    tex.write("\\[\n")

    tex.write("\\begin{aligned}\n")

    tex.write(
        f"{name} &="
    )

    tex.write("\n")

    tex.write(
        "\\begin{pmatrix}\n"
    )

    for i, row in enumerate(rows):

        line = " & ".join(row)

        if i < len(rows) - 1:

            tex.write(
                line
                + r" \\"
                + "\n"
            )

        else:

            tex.write(
                line
                + "\n"
            )

    tex.write(
        "\\end{pmatrix}\n"
    )

    tex.write("\\end{aligned}\n")
    tex.write("\\]\n\n")


# ============================================================
# 11. Christoffel symbols
#
# Only independent components mu <= nu are printed.
# Zero components are omitted.
# ============================================================

def calculate_christoffel(
    metric,
    inverse_metric
):

    Gamma = {}

    for rho in range(4):

        for mu in range(4):

            for nu in range(
                mu,
                4
            ):

                value = 0

                for sigma in range(4):

                    value += (
                        sp.Rational(1, 2)
                        *
                        inverse_metric[
                            rho,
                            sigma
                        ]
                        *
                        (
                            sp.diff(
                                metric[
                                    sigma,
                                    nu
                                ],
                                coords[mu]
                            )
                            +
                            sp.diff(
                                metric[
                                    sigma,
                                    mu
                                ],
                                coords[nu]
                            )
                            -
                            sp.diff(
                                metric[
                                    mu,
                                    nu
                                ],
                                coords[sigma]
                            )
                        )
                    )

                value = simp(
                    value
                )

                if value != 0:

                    Gamma[
                        (
                            rho,
                            mu,
                            nu
                        )
                    ] = value

    return Gamma


# ============================================================
# 12. Automatic tetrad
#
# g_mu_nu = eta_ab e^a_mu e^b_nu
#
# For the diagonal metric:
#
# e^0_0 = sqrt(F)
# e^1_1 = 1/sqrt(F)
# e^2_2 = r
# e^3_3 = r sin(theta)
#
# No expansion of F(r) is performed.
# ============================================================

def automatic_tetrad(
    metric_matrix
):

    if not metric_matrix.is_diagonal():

        raise NotImplementedError(
            "Automatic tetrad construction "
            "currently supports diagonal metrics only."
        )

    tetrad = sp.zeros(
        4,
        4
    )

    tetrad[0, 0] = sp.sqrt(F)

    tetrad[1, 1] = 1 / sp.sqrt(F)

    tetrad[2, 2] = r

    tetrad[3, 3] = (
        r * sp.sin(theta)
    )

    return tetrad.applyfunc(
        simp
    )


# ============================================================
# 13. Spin connection
#
# omega_mu^{ab} =
#
# e^a_nu (
#     partial_mu e^{b nu}
#     + Gamma^nu_mu lambda e^{b lambda}
# )
#
# Only a < b are printed.
# ============================================================

def calculate_spin_connection(
    tetrad_matrix,
    tetrad_inverse,
    christoffel_symbols
):

    omega = {}

    for mu in range(4):

        for a in range(4):

            for b in range(
                a + 1,
                4
            ):

                value = 0

                for nu in range(4):

                    for lam in range(4):

                        value += (
                            tetrad_matrix[
                                a,
                                nu
                            ]
                            *
                            (
                                sp.diff(
                                    tetrad_inverse[
                                        b,
                                        nu
                                    ],
                                    coords[mu]
                                )
                                +
                                christoffel_symbols[
                                    nu,
                                    mu,
                                    lam
                                ]
                                *
                                tetrad_inverse[
                                    b,
                                    lam
                                ]
                            )
                        )

                value = simp(
                    value
                )

                if value != 0:

                    omega[
                        (
                            mu,
                            a,
                            b
                        )
                    ] = value

    return omega


# ============================================================
# 14. Flat-space gamma matrices
#
# Dirac representation
# Signature (+, -, -, -)
# ============================================================

sigma1 = sp.Matrix([
    [0, 1],
    [1, 0]
])

sigma2 = sp.Matrix([
    [0, -sp.I],
    [sp.I, 0]
])

sigma3 = sp.Matrix([
    [1, 0],
    [0, -1]
])

zero2 = sp.zeros(
    2,
    2
)


gamma0 = sp.diag(
    1,
    1,
    -1,
    -1
)


gamma1 = sp.Matrix.vstack(

    sp.Matrix.hstack(
        zero2,
        sigma1
    ),

    sp.Matrix.hstack(
        -sigma1,
        zero2
    )
)


gamma2 = sp.Matrix.vstack(

    sp.Matrix.hstack(
        zero2,
        sigma2
    ),

    sp.Matrix.hstack(
        -sigma2,
        zero2
    )
)


gamma3 = sp.Matrix.vstack(

    sp.Matrix.hstack(
        zero2,
        sigma3
    ),

    sp.Matrix.hstack(
        -sigma3,
        zero2
    )
)


gamma_flat = [
    gamma0,
    gamma1,
    gamma2,
    gamma3
]


# ============================================================
# 15. Spinor connection
#
# Gamma_mu =
# 1/4 omega_mu_ab gamma^a gamma^b
# ============================================================

def calculate_spinor_connection(
    omega_dict
):

    result = []

    for mu in range(4):

        Gamma_mu = sp.zeros(
            4,
            4
        )

        for key, value in omega_dict.items():

            mu0, a, b = key

            if mu0 != mu:
                continue

            omega_lower = (
                eta[a, a]
                *
                eta[b, b]
                *
                value
            )

            Gamma_mu += (
                sp.Rational(1, 4)
                *
                omega_lower
                *
                gamma_flat[a]
                *
                gamma_flat[b]
            )

        result.append(
            Gamma_mu.applyfunc(
                simp
            )
        )

    return result


# ============================================================
# 16. Curved-space gamma matrices
#
# gamma^mu(x) = e^mu_a gamma^a
# ============================================================

def calculate_curved_gamma(
    tetrad_inverse
):

    gamma_curved = []

    for mu in range(4):

        matrix = sp.zeros(
            4,
            4
        )

        for a in range(4):

            matrix += (
                tetrad_inverse[
                    mu,
                    a
                ]
                *
                gamma_flat[a]
            )

        gamma_curved.append(
            matrix.applyfunc(
                simp
            )
        )

    return gamma_curved


# ============================================================
# 17. Dirac spinor
# ============================================================

psi = sp.Matrix([

    sp.Function(
        "psi_1"
    )(*coords),

    sp.Function(
        "psi_2"
    )(*coords),

    sp.Function(
        "psi_3"
    )(*coords),

    sp.Function(
        "psi_4"
    )(*coords)

])


# ============================================================
# 18. Covariant derivative
#
# D_mu psi =
# (partial_mu + Gamma_mu) psi
# ============================================================

def calculate_covariant_derivatives(
    spinor,
    spinor_connection
):

    result = []

    for mu in range(4):

        Dpsi = (
            spinor.diff(
                coords[mu]
            )
            +
            spinor_connection[mu]
            *
            spinor
        )

        result.append(
            Dpsi.applyfunc(
                simp
            )
        )

    return result


# ============================================================
# 19. Inverse metric
# ============================================================

g_inv = (
    g.inv()
    .applyfunc(simp)
)


# ============================================================
# 20. EinsteinPy metric
# ============================================================

metric = MetricTensor(
    g.tolist(),
    coords
)


# ============================================================
# 21. Christoffel symbols from EinsteinPy
# ============================================================

christoffel_tensor = (
    ChristoffelSymbols.from_metric(
        metric
    )
)

christoffel_einsteinpy = (
    christoffel_tensor.tensor()
)


# ============================================================
# 22. Calculate Christoffel symbols
# ============================================================

christoffel = calculate_christoffel(
    g,
    g_inv
)


# ============================================================
# 23. Automatic tetrad
# ============================================================

tetrad = automatic_tetrad(
    g
)


# ============================================================
# 24. Inverse tetrad
# ============================================================

tetrad_inv = (
    tetrad.inv()
    .applyfunc(simp)
)


# ============================================================
# 25. Spin connection
# ============================================================

omega = calculate_spin_connection(
    tetrad,
    tetrad_inv,
    christoffel_einsteinpy
)


# ============================================================
# 26. Spinor connection
# ============================================================

Gamma_spinor = (
    calculate_spinor_connection(
        omega
    )
)


# ============================================================
# 27. Curved gamma matrices
# ============================================================

gamma_curved = (
    calculate_curved_gamma(
        tetrad_inv
    )
)


# ============================================================
# 28. Covariant derivatives
# ============================================================

covariant_psi = (
    calculate_covariant_derivatives(
        psi,
        Gamma_spinor
    )
)


# ============================================================
# 29. Generate LaTeX document
# ============================================================

output_file = (
    "dirac_geometry.tex"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as tex:

    # ========================================================
    # Preamble
    # ========================================================

    tex.write(
r"""\documentclass[12pt,fleqn]{article}

\usepackage[a4paper,margin=1in]{geometry}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{bm}
\usepackage{array}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage{physics}
\usepackage{slashed}

\usepackage{pdflscape}

\setlength{\mathindent}{0pt}

\begin{document}

"""
    )


    # ========================================================
    # Title
    # ========================================================

    tex.write(
r"""\section*{Geometry for the Curved-Space Dirac Equation}

"""
    )


    # ========================================================
    # Coordinates
    # ========================================================

    tex.write(
r"""\section*{Coordinates}

\[
(t,r,\theta,\phi)
\]

"""
    )


    # ========================================================
    # Line element
    # ========================================================

    tex.write(
r"""\section*{Line Element}

\[
\begin{aligned}
ds^2
={}&
F(r)\,dt^2
-\frac{1}{F(r)}\,dr^2
-r^2\,d\theta^2
-r^2\sin^2\theta\,d\phi^2 .
\end{aligned}
\]

"""
    )


    # ========================================================
    # Metric tensor
    # ========================================================

    tex.write(
r"""\section*{Metric Tensor}

"""
    )

    write_matrix(
        tex,
        r"g_{\mu\nu}",
        g
    )


    # ========================================================
    # Inverse metric
    # ========================================================

    tex.write(
r"""\section*{Inverse Metric}

"""
    )

    write_matrix(
        tex,
        r"g^{\mu\nu}",
        g_inv
    )


    # ========================================================
    # Christoffel symbols
    # ========================================================

    tex.write(
r"""\section*{Christoffel Symbols}

The Christoffel symbols are defined by

\[
\Gamma^\rho{}_{\mu\nu}
=
\frac{1}{2}
g^{\rho\sigma}
\left(
\partial_\mu g_{\sigma\nu}
+
\partial_\nu g_{\sigma\mu}
-
\partial_\sigma g_{\mu\nu}
\right).
\]

"""
    )


    for key, value in christoffel.items():

        rho, mu0, nu = key

        tex.write("\\[\n")

        tex.write(
            "\\begin{aligned}\n"
        )

        tex.write(
            rf"\Gamma^{{{rho}}}_{{{mu0}{nu}}}"
            rf" &= {latex_expr(value)}"
        )

        tex.write("\n")

        tex.write(
            "\\end{aligned}\n"
        )

        tex.write("\\]\n\n")


    # ========================================================
    # Tetrad
    # ========================================================

    tex.write(
r"""\section*{Tetrad}

The tetrad is obtained from

\[
g_{\mu\nu}
=
\eta_{ab}
e^a{}_\mu
e^b{}_\nu ,
\qquad
\eta_{ab}
=
\operatorname{diag}(1,-1,-1,-1).
\]

"""
    )

    write_matrix(
        tex,
        r"e^a{}_\mu",
        tetrad
    )


    # ========================================================
    # Inverse tetrad
    # ========================================================

    tex.write(
r"""\section*{Inverse Tetrad}

\[
e^a{}_\mu e^\mu{}_b
=
\delta^a{}_b ,
\qquad
e^a{}_\mu e^\nu{}_a
=
\delta^\nu{}_\mu .
\]

"""
    )

    write_matrix(
        tex,
        r"e^\mu{}_a",
        tetrad_inv
    )


    # ========================================================
    # Spin connection
    # ========================================================

    tex.write(
r"""\section*{Spin Connection}

The spin connection coefficients are

\[
\omega_\mu{}^{ab}
=
e^a{}_\nu
\left(
\partial_\mu e^{b\nu}
+
\Gamma^\nu{}_{\mu\lambda}
e^{b\lambda}
\right).
\]

Only the independent components with \(a<b\) are displayed.

"""
    )


    for (
        mu0,
        a,
        b
    ), value in omega.items():

        tex.write("\\[\n")

        tex.write(
            "\\begin{aligned}\n"
        )

        tex.write(
            rf"\omega_{{{mu0}}}^{{{a}{b}}}"
            rf" &= {latex_expr(value)}"
        )

        tex.write("\n")

        tex.write(
            "\\end{aligned}\n"
        )

        tex.write("\\]\n\n")


    # ========================================================
    # Spinor connection
    # ========================================================

    tex.write(
r"""\section*{Spinor Connection}

\[
\Gamma_\mu
=
\frac{1}{4}
\omega_{\mu ab}
\gamma^a\gamma^b .
\]

"""
    )


    for mu0 in range(4):

        write_matrix(
            tex,
            rf"\Gamma_{{{mu0}}}",
            Gamma_spinor[mu0]
        )


    # ========================================================
    # Flat gamma matrices
    # ========================================================

    tex.write(
r"""\section*{Flat-Space Gamma Matrices}

The flat-space gamma matrices satisfy

\[
\left\{
\gamma^a,
\gamma^b
\right\}
=
2\eta^{ab}I_4 .
\]

"""
    )


    for a, matrix in enumerate(
        gamma_flat
    ):

        write_matrix(
            tex,
            rf"\gamma^{{{a}}}",
            matrix
        )


    # ========================================================
    # Curved gamma matrices
    # ========================================================

    tex.write(
r"""\section*{Curved-Space Gamma Matrices}

The curved-space gamma matrices are

\[
\gamma^\mu(x)
=
e^\mu{}_a(x)
\gamma^a .
\]

"""
    )


    for mu0, matrix in enumerate(
        gamma_curved
    ):

        # A completely zero gamma matrix is explicitly printed.

        if matrix == sp.zeros(
            4,
            4
        ):

            tex.write("\\[\n")

            tex.write(
                "\\begin{aligned}\n"
            )

            tex.write(
                rf"\gamma^{{{mu0}}}(x) &= 0"
            )

            tex.write("\n")

            tex.write(
                "\\end{aligned}\n"
            )

            tex.write("\\]\n\n")

        else:

            write_matrix(
                tex,
                rf"\gamma^{{{mu0}}}(x)",
                matrix
            )


    # ========================================================
    # Covariant derivative
    # ========================================================

    tex.write(
r"""\section*{Covariant Derivative}

For a Dirac spinor,

\[
D_\mu\psi
=
\left(
\partial_\mu+\Gamma_\mu
\right)\psi .
\]

"""
    )


    for mu0, Dpsi in enumerate(
        covariant_psi
    ):

        tex.write("\\[\n")

        tex.write(
            "\\begin{aligned}\n"
        )

        tex.write(
            rf"D_{{{mu0}}}\psi"
            rf" &="
        )

        tex.write("\n")

        tex.write(
            "\\begin{pmatrix}\n"
        )

        for i in range(
            Dpsi.rows
        ):

            value = latex_expr(
                Dpsi[i]
            )

            if i < Dpsi.rows - 1:

                tex.write(
                    value
                    + r" \\"
                    + "\n"
                )

            else:

                tex.write(
                    value
                    + "\n"
                )

        tex.write(
            "\\end{pmatrix}\n"
        )

        tex.write(
            "\\end{aligned}\n"
        )

        tex.write("\\]\n\n")


    # ========================================================
    # Final Dirac equation
    # ========================================================

    tex.write(
r"""\section*{Curved-Space Dirac Equation}

The curved-space Dirac equation is

\[
\boxed{
\left[
i\gamma^\mu(x)
\left(
\partial_\mu+\Gamma_\mu
\right)
-m
\right]
\psi(x)
=
0
}
\]

"""
    )


    # ========================================================
    # End document
    # ========================================================

    tex.write(
r"""\end{document}
"""
    )


print(
    f"LaTeX file generated successfully: {output_file}"
)