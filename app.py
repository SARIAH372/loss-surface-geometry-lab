import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loss Surface Geometry Lab", layout="wide")

# ============================
# Core math
# ============================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def make_dataset(kind, n, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 2))

    if kind == "Classification (logistic)":
        w_true = np.array([1.2, -1.0])
        logits = X @ w_true
        p = sigmoid(logits)
        y = (p > 0.5).astype(float)
        flip = rng.random(n) < noise
        y = np.where(flip, 1.0 - y, y)
        return X, y

    # Regression (MSE)
    w_true = np.array([1.0, -0.8])
    y = X @ w_true + rng.normal(0, noise, size=n)
    return X, y

def loss_and_grad(kind, X, y, w, reg):
    z = X @ w
    if kind == "Classification (logistic)":
        p = sigmoid(z)
        eps = 1e-9
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        grad = (X.T @ (p - y)) / X.shape[0]
    else:
        r = z - y
        loss = 0.5 * np.mean(r**2)
        grad = (X.T @ r) / X.shape[0]

    loss = loss + 0.5 * reg * float(w @ w)
    grad = grad + reg * w
    return float(loss), grad

def hessian(kind, X, y, w, reg):
    n = X.shape[0]
    if kind == "Classification (logistic)":
        p = sigmoid(X @ w)
        s = p * (1 - p)
        H = (X.T * s) @ X / n
    else:
        H = (X.T @ X) / n
    return H + reg * np.eye(2)

# ============================
# Synthetic saddle option
# ============================
def saddle_loss_grad_hess(w, reg, alpha=1.0):
    w1, w2 = float(w[0]), float(w[1])
    f = 0.5 * (w1**2 - alpha * w2**2) + 0.5 * reg * (w1**2 + w2**2)
    g = np.array([w1, -alpha * w2]) + reg * np.array([w1, w2])
    H = np.array([[1.0, 0.0], [0.0, -alpha]]) + reg * np.eye(2)
    return float(f), g, H

# ============================
# Surfaces + fields
# ============================
def loss_grid(get_fgh, w1s, w2s):
    L = np.zeros((len(w2s), len(w1s)), dtype=float)
    for i, w2 in enumerate(w2s):
        for j, w1 in enumerate(w1s):
            f, _, _ = get_fgh(np.array([w1, w2], dtype=float))
            L[i, j] = f
    return L

def hessian_eigs_grid(get_fgh, w1s, w2s):
    e1 = np.zeros((len(w2s), len(w1s)), dtype=float)
    e2 = np.zeros((len(w2s), len(w1s)), dtype=float)
    for i, w2 in enumerate(w2s):
        for j, w1 in enumerate(w1s):
            _, _, H = get_fgh(np.array([w1, w2], dtype=float))
            vals = np.linalg.eigvalsh(H)
            e1[i, j], e2[i, j] = float(vals[0]), float(vals[1])
    return e1, e2

def gradient_field(get_fgh, R, coarse=22):
    gw1s = np.linspace(-R, R, coarse)
    gw2s = np.linspace(-R, R, coarse)
    Gx = np.zeros((coarse, coarse), dtype=float)
    Gy = np.zeros((coarse, coarse), dtype=float)
    for i, w2 in enumerate(gw2s):
        for j, w1 in enumerate(gw1s):
            _, g, _ = get_fgh(np.array([w1, w2], dtype=float))
            Gx[i, j] = -g[0]
            Gy[i, j] = -g[1]
    return gw1s, gw2s, Gx, Gy

# ============================
# Optimization building blocks
# ============================
def line_search_grid(loss_fn, w, d, alphas):
    best_a = float(alphas[0])
    best_f = None
    for a in alphas:
        f = loss_fn(w + a * d)
        if (best_f is None) or (f < best_f):
            best_f = f
            best_a = float(a)
    return best_a, float(best_f)

def trust_region_step(H, g, delta, damping=1e-8):
    Hs = H + damping * np.eye(2)
    try:
        p0 = -np.linalg.solve(Hs, g)
    except np.linalg.LinAlgError:
        ng = np.linalg.norm(g) + 1e-12
        return -delta * g / ng

    if np.linalg.norm(p0) <= delta:
        return p0

    def p_of(lam):
        return -np.linalg.solve(Hs + lam * np.eye(2), g)

    lam_lo, lam_hi = 0.0, 1.0
    for _ in range(60):
        if np.linalg.norm(p_of(lam_hi)) <= delta:
            break
        lam_hi *= 2.0

    for _ in range(60):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if np.linalg.norm(p_of(lam_mid)) > delta:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return p_of(lam_hi)

def implicit_gd_step_quadratic(H, g, eta, damping=1e-8):
    A = np.eye(2) + eta * (H + damping * np.eye(2))
    return -np.linalg.solve(A, eta * g)

def run_optim(get_fgh, steps, w_init, method, eta, momentum,
             use_line_search, ls_max, ls_grid,
             newton_damping, tr_delta, implicit_eta):
    w = w_init.astype(float).copy()
    v = np.zeros_like(w)

    path = [w.copy()]
    losses, grad_norms, min_eigs, max_eigs = [], [], [], []

    def f_only(wv):
        f, _, _ = get_fgh(wv)
        return f

    for _ in range(steps):
        f, g, H = get_fgh(w)
        eigs = np.linalg.eigvalsh(H)

        losses.append(float(f))
        grad_norms.append(float(np.linalg.norm(g)))
        min_eigs.append(float(eigs[0]))
        max_eigs.append(float(eigs[1]))

        if method == "Newton (damped)":
            Hd = H + float(newton_damping) * np.eye(2)
            d = -np.linalg.solve(Hd, g)
            if use_line_search:
                alphas = np.linspace(0.0, ls_max, ls_grid)
                a, _ = line_search_grid(f_only, w, d, alphas)
                w = w + a * d
            else:
                w = w + eta * d

        elif method == "Trust region (Newton model)":
            p = trust_region_step(H, g, delta=tr_delta, damping=float(newton_damping))
            w = w + p

        elif method == "Implicit gradient step":
            p = implicit_gd_step_quadratic(H, g, eta=float(implicit_eta), damping=float(newton_damping))
            w = w + p

        else:
            if use_line_search:
                d = -g
                alphas = np.linspace(0.0, ls_max, ls_grid)
                a, _ = line_search_grid(f_only, w, d, alphas)
                w = w + a * d
            else:
                v = momentum * v + g
                w = w - eta * v

        path.append(w.copy())

    return (np.array(path),
            np.array(losses),
            np.array(grad_norms),
            np.array(min_eigs),
            np.array(max_eigs))

def run_adam(get_fgh, steps, w_init, eta, beta1=0.9, beta2=0.999, eps=1e-8):
    w = w_init.astype(float).copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    path = [w.copy()]
    losses, grad_norms, min_eigs, max_eigs = [], [], [], []

    for t in range(1, steps + 1):
        f, g, H = get_fgh(w)
        eigs = np.linalg.eigvalsh(H)

        losses.append(float(f))
        grad_norms.append(float(np.linalg.norm(g)))
        min_eigs.append(float(eigs[0]))
        max_eigs.append(float(eigs[1]))

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)

        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)

        w = w - eta * mhat / (np.sqrt(vhat) + eps)
        path.append(w.copy())

    return (np.array(path),
            np.array(losses),
            np.array(grad_norms),
            np.array(min_eigs),
            np.array(max_eigs))

def stability_map(get_fgh, w_init, etas, steps=80, diverge_radius=25.0):
    out = np.zeros(len(etas), dtype=int)
    for i, eta in enumerate(etas):
        w = w_init.astype(float).copy()
        prev_f = None
        bad = 0
        diverged = False
        for _ in range(steps):
            f, g, _ = get_fgh(w)
            if prev_f is not None and f > prev_f + 1e-6:
                bad += 1
            prev_f = f
            w = w - eta * g
            if np.linalg.norm(w) > diverge_radius or (not np.isfinite(w).all()) or (not np.isfinite(f)):
                diverged = True
                break
        if diverged:
            out[i] = 2
        else:
            out[i] = 1 if bad > steps * 0.25 else 0
    return out

# ============================
# Presets
# ============================
PRESETS = {
    "Saddle (negative curvature)": dict(
        mode="Saddle-point (synthetic)",
        alpha=2.0,
        reg=0.01,
        method="Gradient descent / Momentum",
        eta=0.15,
        momentum=0.0,
        use_line_search=False,
        steps=120,
        w1_init=-2.0,
        w2_init=2.0,
        heat_choice="min eigenvalue",
        w_range=3.0,
        grid=140,
    ),
    "Sharp valley (ill-conditioned)": dict(
        mode="Data-driven loss",
        kind="Regression (MSE)",
        n=220,
        noise=0.05,
        seed=7,
        reg=0.02,
        method="Gradient descent / Momentum",
        eta=0.25,
        momentum=0.90,
        use_line_search=False,
        steps=140,
        w1_init=-3.0,
        w2_init=3.0,
        heat_choice="condition number (|max/min|)",
        w_range=4.0,
        grid=150,
    ),
    "Flat basin (slow gradients)": dict(
        mode="Data-driven loss",
        kind="Regression (MSE)",
        n=140,
        noise=0.25,
        seed=7,
        reg=0.80,
        method="Gradient descent / Momentum",
        eta=0.10,
        momentum=0.0,
        use_line_search=False,
        steps=160,
        w1_init=-2.0,
        w2_init=2.0,
        heat_choice="max eigenvalue",
        w_range=3.0,
        grid=140,
    ),
    "Logistic (noisy labels)": dict(
        mode="Data-driven loss",
        kind="Classification (logistic)",
        n=260,
        noise=0.25,
        seed=9,
        reg=0.08,
        method="Adam",
        eta=0.12,
        momentum=0.0,
        use_line_search=False,
        steps=170,
        w1_init=-2.5,
        w2_init=2.5,
        heat_choice="max eigenvalue",
        w_range=4.0,
        grid=150,
    ),
}

def apply_preset(name: str):
    cfg = PRESETS[name]
    for k, v in cfg.items():
        st.session_state[k] = v

# ============================
# UI
# ============================
st.title("🧠 Loss Surface Geometry Lab")
st.caption("A calculus-driven ML lab: landscapes, gradients, curvature, and optimizer dynamics in 2D weight space.")

with st.sidebar:
    st.header("Presets")
    preset = st.selectbox("Load a preset", ["(none)"] + list(PRESETS.keys()), key="preset")
    if preset != "(none)":
        apply_preset(preset)

    st.header("Landscape")
    mode = st.selectbox(
        "Source",
        ["Data-driven loss", "Saddle-point (synthetic)"],
        index=0,
        key="mode"
    )

    st.header("Surface window")
    w_range = st.slider("Weight range (±R)", 0.5, 8.0, st.session_state.get("w_range", 3.0), 0.1, key="w_range")
    grid = st.slider("Grid resolution", 40, 220, int(st.session_state.get("grid", 130)), 10, key="grid")

    st.header("Regularization")
    reg = st.slider("L2 regularization (λ)", 0.0, 3.0, float(st.session_state.get("reg", 0.10)), 0.01, key="reg")

    if mode == "Data-driven loss":
        st.header("Dataset")
        kind = st.selectbox(
            "Task",
            ["Classification (logistic)", "Regression (MSE)"],
            index=0 if st.session_state.get("kind", "Classification (logistic)") == "Classification (logistic)" else 1,
            key="kind"
        )
        n = st.slider("Number of points", 20, 500, int(st.session_state.get("n", 140)), 10, key="n")
        noise = st.slider("Noise / label-flip", 0.0, 0.6, float(st.session_state.get("noise", 0.10)), 0.01, key="noise")
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=int(st.session_state.get("seed", 7)), step=1, key="seed")
    else:
        st.header("Saddle parameters")
        alpha = st.slider("Saddle curvature α (negative in w2)", 0.2, 5.0, float(st.session_state.get("alpha", 1.5)), 0.1, key="alpha")

    st.header("Optimization")
    method = st.selectbox(
        "Method",
        ["Gradient descent / Momentum", "Newton (damped)", "Trust region (Newton model)", "Implicit gradient step", "Adam"],
        index=0,
        key="method"
    )

    eta = st.slider("Step scale (η)", 0.001, 1.5, float(st.session_state.get("eta", 0.08)), 0.001, key="eta")
    momentum = st.slider("Momentum (GD only)", 0.0, 0.99, float(st.session_state.get("momentum", 0.0)), 0.01, key="momentum")
    steps = st.slider("Steps", 10, 350, int(st.session_state.get("steps", 90)), 5, key="steps")

    st.subheader("Initialization")
    w1_init = st.slider("Init w1", -8.0, 8.0, float(st.session_state.get("w1_init", -2.0)), 0.1, key="w1_init")
    w2_init = st.slider("Init w2", -8.0, 8.0, float(st.session_state.get("w2_init", 2.0)), 0.1, key="w2_init")

    st.subheader("Line search")
    use_line_search = st.toggle("Use 1D line-search (GD/Newton only)", value=bool(st.session_state.get("use_line_search", False)), key="use_line_search")
    ls_max = st.slider("Line-search max α", 0.2, 6.0, float(st.session_state.get("ls_max", 2.0)), 0.1, key="ls_max")
    ls_grid = st.slider("Line-search grid points", 20, 200, int(st.session_state.get("ls_grid", 80)), 5, key="ls_grid")

    st.subheader("Curvature controls")
    newton_damping = st.slider("Damping ε (Newton/TR/Implicit)", 0.0, 0.2, float(st.session_state.get("newton_damping", 0.01)), 0.001, key="newton_damping")

    st.subheader("Trust region")
    tr_delta = st.slider("Trust region radius Δ", 0.05, 3.0, float(st.session_state.get("tr_delta", 0.6)), 0.01, key="tr_delta")

    st.subheader("Implicit step")
    implicit_eta = st.slider("Implicit η (backward Euler)", 0.001, 2.0, float(st.session_state.get("implicit_eta", 0.25)), 0.001, key="implicit_eta")

    st.header("Heatmap")
    heat_choice = st.selectbox(
        "Quantity",
        ["min eigenvalue", "max eigenvalue", "condition number (|max/min|)"],
        index=0,
        key="heat_choice"
    )

    st.header("Local geometry")
    show_local_vectors = st.toggle("Show gradient + tangent at final point", value=bool(st.session_state.get("show_local_vectors", True)), key="show_local_vectors")
    local_vector_scale = st.slider("Local vector scale", 0.1, 3.0, float(st.session_state.get("local_vector_scale", 1.0)), 0.1, key="local_vector_scale")

    st.header("Flow visuals")
    show_streamlines = st.toggle("Show streamlines on landscape", value=bool(st.session_state.get("show_streamlines", True)), key="show_streamlines")
    streamline_density = st.slider("Streamline density", 0.3, 2.5, float(st.session_state.get("streamline_density", 1.0)), 0.1, key="streamline_density")

# ============================
# Build get_fgh
# ============================
w1s = np.linspace(-w_range, w_range, int(grid))
w2s = np.linspace(-w_range, w_range, int(grid))
w_init = np.array([w1_init, w2_init], dtype=float)

if mode == "Data-driven loss":
    X, y = make_dataset(kind, int(n), float(noise), int(seed))

    def get_fgh(w):
        f, g = loss_and_grad(kind, X, y, w, float(reg))
        H = hessian(kind, X, y, w, float(reg))
        return f, g, H
else:
    def get_fgh(w):
        return saddle_loss_grad_hess(w, float(reg), alpha=float(alpha))

# Surfaces
L = loss_grid(get_fgh, w1s, w2s)
e1, e2 = hessian_eigs_grid(get_fgh, w1s, w2s)

# Heatmap selection
heat_choice_val = st.session_state.get("heat_choice", "min eigenvalue")
if heat_choice_val == "min eigenvalue":
    Hmap = e1
    htitle = "Hessian min eigenvalue (negative → saddle direction)"
elif heat_choice_val == "max eigenvalue":
    Hmap = e2
    htitle = "Hessian max eigenvalue (sharpness)"
else:
    denom = np.where(np.abs(e1) < 1e-9, np.nan, np.abs(e1))
    Hmap = np.abs(e2) / denom
    htitle = "Condition proxy |λ_max / λ_min|"

# Gradient field (for quiver + streamlines)
gw1s, gw2s, Gx, Gy = gradient_field(get_fgh, float(w_range), coarse=25)
W1c, W2c = np.meshgrid(gw1s, gw2s)

# Main run
if method == "Adam":
    path, losses, grad_norms, min_eigs, max_eigs = run_adam(
        get_fgh=get_fgh,
        steps=int(steps),
        w_init=w_init,
        eta=float(eta),
    )
else:
    path, losses, grad_norms, min_eigs, max_eigs = run_optim(
        get_fgh=get_fgh,
        steps=int(steps),
        w_init=w_init,
        method=method,
        eta=float(eta),
        momentum=float(momentum),
        use_line_search=bool(use_line_search),
        ls_max=float(ls_max),
        ls_grid=int(ls_grid),
        newton_damping=float(newton_damping),
        tr_delta=float(tr_delta),
        implicit_eta=float(implicit_eta),
    )

w_last = path[-1]
f_last, g_last, _ = get_fgh(w_last)
g_norm = float(np.linalg.norm(g_last) + 1e-12)
grad_dir = -g_last / g_norm
tangent_dir = np.array([-grad_dir[1], grad_dir[0]])

# ============================
# Export data (final polish)
# ============================
def make_export_csv(path, losses, grad_norms, min_eigs, max_eigs):
    steps_used = len(losses)
    rows = ["step,w1,w2,loss,grad_norm,min_eig,max_eig"]
    for t in range(steps_used):
        w = path[t]  # aligns with losses[t]
        rows.append(
            f"{t},{w[0]:.10g},{w[1]:.10g},{losses[t]:.10g},{grad_norms[t]:.10g},{min_eigs[t]:.10g},{max_eigs[t]:.10g}"
        )
    return "\n".join(rows).encode("utf-8")

export_csv_bytes = make_export_csv(path, losses, grad_norms, min_eigs, max_eigs)

# ============================
# Tabs
# ============================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Landscape",
    "Curvature",
    "Optimizer path",
    "Compare methods",
    "Stability map",
    "Export"
])

with tab1:
    st.subheader("Loss contours + gradient field + path")

    fig, ax = plt.subplots()
    cs = ax.contour(w1s, w2s, L, levels=26)
    ax.clabel(cs, inline=True, fontsize=8)

    ax.quiver(W1c, W2c, Gx, Gy, angles="xy", scale_units="xy", scale=1.0, width=0.0025)

    if show_streamlines:
        ax.streamplot(W1c, W2c, Gx, Gy, density=float(streamline_density), linewidth=0.8, arrowsize=0.8)

    ax.plot(path[:, 0], path[:, 1], marker="o", markersize=3, linewidth=1.6)
    ax.scatter([w_init[0]], [w_init[1]], s=90, marker="X", label="init")
    ax.scatter([w_last[0]], [w_last[1]], s=90, label="final")

    if show_local_vectors:
        s = float(local_vector_scale)
        ax.arrow(w_last[0], w_last[1], s * grad_dir[0], s * grad_dir[1],
                 head_width=0.12, length_includes_head=True)
        ax.arrow(w_last[0], w_last[1], s * tangent_dir[0], s * tangent_dir[1],
                 head_width=0.12, length_includes_head=True)

    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_xlim([-w_range, w_range])
    ax.set_ylim([-w_range, w_range])
    ax.set_title("Loss landscape in (w1, w2)")
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
**Local geometry overlay (final point):**
- Arrow 1: downhill direction (−∇L)  
- Arrow 2: tangent direction (orthogonal to ∇L; along level sets)
"""
    )

with tab2:
    st.subheader("Hessian eigen-geometry heatmap + eigenvalues along the path")

    figH, axH = plt.subplots()
    im = axH.imshow(
        Hmap,
        extent=[w1s.min(), w1s.max(), w2s.min(), w2s.max()],
        origin="lower",
        aspect="auto"
    )
    axH.set_xlabel("w1")
    axH.set_ylabel("w2")
    axH.set_title(htitle)
    plt.colorbar(im, ax=axH, fraction=0.046, pad=0.04)

    axH.plot(path[:, 0], path[:, 1], linewidth=1.4)
    axH.scatter([w_init[0]], [w_init[1]], s=70, marker="X")
    axH.scatter([w_last[0]], [w_last[1]], s=70)
    st.pyplot(figH, clear_figure=True)

    figE, axE = plt.subplots()
    axE.plot(np.arange(len(min_eigs)), min_eigs, label="min eig(H)")
    axE.plot(np.arange(len(max_eigs)), max_eigs, label="max eig(H)")
    axE.axhline(0.0, linewidth=1.0)
    axE.set_xlabel("step")
    axE.set_ylabel("eigenvalue")
    axE.set_title("Hessian eigenvalues along the optimization path")
    axE.legend()
    st.pyplot(figE, clear_figure=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final loss", f"{losses[-1]:.4f}")
    c2.metric("Final ||∇L||", f"{grad_norms[-1]:.4f}")
    c3.metric("Final min eig(H)", f"{min_eigs[-1]:.4f}")
    c4.metric("Final max eig(H)", f"{max_eigs[-1]:.4f}")

with tab3:
    st.subheader("Optimization diagnostics")

    lcol, rcol = st.columns(2)
    with lcol:
        fig2, ax2 = plt.subplots()
        ax2.plot(np.arange(len(losses)), losses)
        ax2.set_xlabel("step")
        ax2.set_ylabel("loss")
        ax2.set_title("Loss vs step")
        st.pyplot(fig2, clear_figure=True)

    with rcol:
        fig3, ax3 = plt.subplots()
        ax3.plot(np.arange(len(grad_norms)), grad_norms)
        ax3.set_xlabel("step")
        ax3.set_ylabel("||grad||")
        ax3.set_title("Gradient norm vs step")
        st.pyplot(fig3, clear_figure=True)

    st.markdown(
        """
Try:
- Increase **η** to push into unstable regions.  
- Switch **method** to see curvature-aware vs gradient-only behavior.  
"""
    )

with tab4:
    st.subheader("Compare methods (side-by-side paths)")
    st.caption("Runs multiple optimizers with the same initialization and overlays their paths.")

    compare = st.multiselect(
        "Methods to compare",
        ["Gradient descent / Momentum", "Newton (damped)", "Trust region (Newton model)", "Implicit gradient step", "Adam"],
        default=["Gradient descent / Momentum", "Newton (damped)", "Trust region (Newton model)"]
    )

    paths = {}
    for m in compare:
        if m == "Adam":
            pth, _, _, _, _ = run_adam(get_fgh, steps=int(steps), w_init=w_init, eta=float(eta))
        else:
            pth, _, _, _, _ = run_optim(
                get_fgh=get_fgh,
                steps=int(steps),
                w_init=w_init,
                method=m,
                eta=float(eta),
                momentum=float(momentum),
                use_line_search=bool(use_line_search),
                ls_max=float(ls_max),
                ls_grid=int(ls_grid),
                newton_damping=float(newton_damping),
                tr_delta=float(tr_delta),
                implicit_eta=float(implicit_eta),
            )
        paths[m] = pth

    figC, axC = plt.subplots()
    cs = axC.contour(w1s, w2s, L, levels=22)
    axC.clabel(cs, inline=True, fontsize=8)

    for name, pth in paths.items():
        axC.plot(pth[:, 0], pth[:, 1], linewidth=1.8, label=name)

    axC.scatter([w_init[0]], [w_init[1]], s=90, marker="X", label="init")
    axC.set_xlabel("w1")
    axC.set_ylabel("w2")
    axC.set_xlim([-w_range, w_range])
    axC.set_ylim([-w_range, w_range])
    axC.set_title("Overlay: optimizer paths on the same landscape")
    axC.legend(loc="best", fontsize=8)
    st.pyplot(figC, clear_figure=True)

    st.markdown(
        """
**Instant demos:** load a preset (sidebar) and then compare methods.
- *Saddle (negative curvature)* → see why **min eig(H) < 0** matters  
- *Sharp valley* → see conditioning effects on paths  
"""
    )

with tab5:
    st.subheader("Stability map (GD baseline)")
    st.caption("A step-size phase diagram using plain GD: 0=converged-ish, 1=oscillatory/slow, 2=diverged.")

    eta_min = st.slider("η min", 0.001, 1.0, 0.01, 0.001, key="stab_eta_min")
    eta_max = st.slider("η max", 0.01, 3.0, 1.0, 0.01, key="stab_eta_max")
    eta_points = st.slider("η points", 20, 200, 80, 5, key="stab_eta_points")
    stab_steps = st.slider("stability steps", 20, 200, 80, 5, key="stab_steps")

    etas = np.linspace(float(eta_min), float(eta_max), int(eta_points))
    status = stability_map(get_fgh, w_init=w_init, etas=etas, steps=int(stab_steps))

    figS, axS = plt.subplots()
    axS.plot(etas, status)
    axS.set_xlabel("η")
    axS.set_ylabel("status (0 ok, 1 shaky, 2 diverged)")
    axS.set_title("Stability vs step size (GD)")
    st.pyplot(figS, clear_figure=True)

    st.write("Tip: load **Saddle (negative curvature)** preset and increase η_max to see divergence regions quickly.")

with tab6:
    st.subheader("Export (portfolio-ready)")

    st.markdown(
        """
Download the full optimization trace as CSV:
- `step`
- `(w1, w2)`
- `loss`
- `||grad||`
- `min_eig(H)`, `max_eig(H)`
"""
    )

    fname = f"trace_{mode.replace(' ','_')}_{method.replace(' ','_')}.csv"
    st.download_button(
        "Download optimization trace CSV",
        data=export_csv_bytes,
        file_name=fname,
        mime="text/csv",
    )

    st.markdown("You can plot these columns later (or include them in your portfolio writeup).")

    # Tiny preview table (first ~10 rows)
    st.write("Preview (first 10 rows):")
    preview_lines = export_csv_bytes.decode("utf-8").splitlines()[:11]
    st.code("\n".join(preview_lines), language="text")

        