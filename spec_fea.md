# spec_fea.md — FEA version of the hydrogel plate PINN (for Claude Code)

Build this in a NEW folder `plate_fea`, next to `plate_pinn_rq_all_overnight`.
Do NOT edit any PINN folder. You may **import** from the PINN code (read-only) and reuse `config.yaml`, `data/<case>.npz`, `extract_abaqus_modes.py` outputs.
Keep the same staged workflow and the same output format as the PINN, so results can be compared line by line.

---

## 0. Goal in plain words
- Solve the **same plate vibration problem** as the PINN (v0.3, method rayleigh) with the Finite Element Method.
- FE idea: split the plate into small rectangles → build stiffness matrix **K** and mass matrix **M** from the same energies U and T → solve **K φ = ω² M φ** → ω gives frequency, φ gives mode shape.
- No training, no seeds. A full 5-mode run takes **seconds**, not minutes.
- Two FE models:

| Model | Name in code | Theory | Purpose |
| --- | --- | --- | --- |
| A | `senjanovic` | Senjanović two-field Mindlin (w_b, w_s) — **exactly the PINN energy** | Verify the PINN (should match PINN/Ritz to ~0.1 %) |
| B | `mindlin3` | Classical 3-field Mindlin (w, ψx, ψy) — what shell FE codes (Abaqus S4R) use | Check the theory choice; sits between PINN and Abaqus 3D |

- Material: same linearised Ogden as PINN → G = μ1 = 60012.1349 Pa, ν = 0.5, E = 3G, ρ = 1300 kg/m³, k = 5/6. α1 is not used (small vibration about undeformed state).
- **Scope limit (important):** this FEA is linear (small amplitude, no pre-stress), same as the PINN. It is NOT a nonlinear hyperelastic FE. Pre-stress / NLGEOM would need a geometric stiffness matrix — out of scope.

---

## 1. Folder and files
```
plate_fea/
  config.yaml                 # copy of PINN config.yaml + new 'fea:' section (section 2)
  1_preprocess.py             # copy from PINN (unchanged) OR read existing data/<case>.npz
  2_run_fea.py                # NEW: build K, M, apply BCs, solve, save
  3_postprocess_fea.py        # NEW: contour/3D plots of W, frequency table
  4_compare_all.py            # NEW: PINN vs FEA-A vs FEA-B vs Ritz vs Abaqus, + MAC
  run_all_fea.py              # NEW: loop over 4 BCs x 2 models x meshes (mesh study)
  fea/
    mesh.py                   # structured rectangular mesh, node numbering, edge node lists
    hermite.py                # 1D Hermite cubics and the BFS 2D shape functions (Model A)
    element_senjanovic.py     # element K, M for Model A
    element_mitc4.py          # element K, M for Model B
    assemble.py               # sparse global assembly (scipy.sparse COO -> CSR)
    bcs.py                    # edge DOFs to fix for each edge type + clamp_type
    solve.py                  # eigsh shift-invert, sorting, mass normalisation
    evaluate.py               # mode shape W(x,y) on any grid (101 x 51 like the PINN)
    checks.py                 # (m,n) estimate, order check, orthogonality, MAC
  tests/
    test_ssss_exact.py        # closed-form Mindlin SSSS check (section 8.1)
    test_thin_limit.py        # Kirchhoff limit check (section 8.2)
    test_patch.py             # rigid body / constant-strain checks of elements
```
Before coding: open the PINN `config.yaml`, `1_preprocess.py`, `pinn/physics.py`, `pinn/rq_trainer.py`, `3_postprocess.py`, `4_compare_abaqus.py` and reuse their key names, constants (`constants_json`: D, m, a, r, s …), `edges_json`, BC names and the (m,n) sign-change function. Do not re-derive constants differently.

Libraries: Python 3.12, NumPy, SciPy (sparse + eigsh), Matplotlib, PyYAML. **float64 everywhere** (FE needs it, and it is cheap on CPU). No GPU needed.

---

## 2. Config additions (config.yaml)
```yaml
fea:
  model: senjanovic           # senjanovic (Model A) | mindlin3 (Model B)
  mesh: [40, 20]              # elements in x (length a) and y (width b); keep square elements (a/b = 2)
  convergence_meshes: [[10,5],[20,10],[40,20],[80,40],[160,80]]
  n_modes: 5                  # modes reported
  n_extra: 5                  # extra modes solved (for Abaqus pairing, close modes)
  gauss_A: 4                  # Gauss points per direction, Model A
  gauss_B: 2                  # Gauss points per direction, Model B (MITC4)
  shear_factor: null          # null = use k from material; or 0.822 / 0.882 for k-sensitivity
# reused from PINN config: geometry, material, boundary, edges, clamp_type (full | senjanovic)
```
Command line:
```
python 2_run_fea.py --case case_05                         # uses config
python 2_run_fea.py --case case_05 --model mindlin3 --mesh 160 80
python run_all_fea.py --case case_05                       # all BCs, both models, all meshes
```

---

## 3. Physics (physical SI units; same as PINN report section 3.2)
Constants: D = E h³ / (12(1−ν²)), S = k G h, m = ρ h, J = ρ h³ / 12.
After solving, also report λ = ω² m a⁴ / D (same as PINN) and f = ω / 2π.

### Model A — Senjanović two-field (same energy as PINN)
Unknown fields: w_b (bending deflection), w_s (shear deflection). w = w_b + w_s, ψx = w_b,x, ψy = w_b,y.
- U = ½ ∫ { D [ w_b,xx² + w_b,yy² + 2ν w_b,xx w_b,yy + 2(1−ν) w_b,xy² ] + S ( w_s,x² + w_s,y² ) } dA
- T/ω² = ½ ∫ { m (w_b + w_s)² + J ( w_b,x² + w_b,y² ) } dA
- Bending energy has 2nd derivatives → w_b needs a **C1 element**. Use the **Bogner–Fox–Schmit (BFS)** rectangle (exact for a rectangular plate).
- No shear locking: shear energy is only in w_s (no coupling term in K).

### Model B — classical 3-field Mindlin
Unknown fields: w, ψx, ψy (convention u = −z ψx, v = −z ψy; γxz = w,x − ψx, γyz = w,y − ψy).
- U = ½ ∫ { D [ ψx,x² + ψy,y² + 2ν ψx,x ψy,y + (1−ν)/2 (ψx,y + ψy,x)² ] + S [ (w,x − ψx)² + (w,y − ψy)² ] } dA
- T/ω² = ½ ∫ { m w² + J (ψx² + ψy²) } dA
- Check: putting ψ = ∇w_b, γ = ∇w_s into Model B gives Model A. So Model A = Model B with the extra constraint "rotation field is a gradient". This constraint makes Model A **stiffer** (higher f), see section 9.
- Plain bilinear Q4 locks in shear → use **MITC4** (Bathe–Dvorkin assumed shear strain).

---

## 4. Elements

### 4.1 Model A: BFS element (16 DOF per field, 32 DOF per element)
- Node DOFs per field: [f, f,x, f,y, f,xy]. Node DOFs total: 8 = [w_b, w_b,x, w_b,y, w_b,xy, w_s, w_s,x, w_s,y, w_s,xy].
- 1D Hermite cubics on t ∈ [0,1], element length L (x = x0 + tL):
  - H1 = 1 − 3t² + 2t³, H2 = L(t − 2t² + t³), H3 = 3t² − 2t³, H4 = L(−t² + t³)
  - Derivatives in x: divide d/dt by L (2nd derivative by L²).
- 2D shape functions (tensor product), for node with x-end i and y-end j:
  value: Hv_i(x)·Hv_j(y);  f,x: Hd_i(x)·Hv_j(y);  f,y: Hv_i(x)·Hd_j(y);  f,xy: Hd_i(x)·Hd_j(y)
  (Hv = H1 or H3, Hd = H2 or H4 for the matching end).
- Element matrices (N = row of 16 shape functions; Gauss 4 × 4):
  - Bb = [N,xx ; N,yy ; 2 N,xy],  Db = D · [[1, ν, 0], [ν, 1, 0], [0, 0, (1−ν)/2]]
  - K_bb = ∫ Bbᵀ Db Bb dA,  K_ss = ∫ S (N,xᵀN,x + N,yᵀN,y) dA,  K_bs = 0
  - M_mm = ∫ m NᵀN dA,  M_J = ∫ J (N,xᵀN,x + N,yᵀN,y) dA
  - M_bb = M_mm + M_J,  M_bs = M_sb = M_mm,  M_ss = M_mm   (mass couples w_b and w_s because w = w_b + w_s)
- Element K = [[K_bb, 0], [0, K_ss]], M = [[M_bb, M_bs], [M_sb, M_ss]].
- All elements are identical on a uniform mesh → compute once, assemble many times (fast).

### 4.2 Model B: MITC4 element (3 DOF per node, 12 per element)
- Bilinear shape functions N_i on natural coords (ξ, η) ∈ [−1,1]²; rectangle → Jacobian diagonal: ∂/∂x = (2/hx) ∂/∂ξ, ∂/∂y = (2/hy) ∂/∂η.
- Bending: κ = [ψx,x ; ψy,y ; ψx,y + ψy,x], 2 × 2 Gauss.
- Shear (MITC4 tying): compute direct γ = ∇w − ψ at edge mid-points, then interpolate
  - γxz(ξ,η) = ½(1+η) γxz(0,+1) + ½(1−η) γxz(0,−1)
  - γyz(ξ,η) = ½(1+ξ) γyz(+1,0) + ½(1−ξ) γyz(−1,0)
  - integrate S·γᵀγ with 2 × 2 Gauss.
- Consistent mass: diag(m, J, J), 2 × 2 Gauss.
- MITC4 converges as O(h²) → needs a finer mesh than Model A (use up to 160 × 80).

### 4.3 Assembly
- Structured mesh, node id = j·(nx+1) + i. Global DOF = node_id·(DOF per node) + local index.
- Build COO (rows, cols, values) with NumPy, convert to CSR. No Python loops over Gauss points inside the element loop.

---

## 5. Boundary conditions (edge names exactly as PINN: x0, xa, y0, yb)
| boundary | x0 | xa | y0 | yb |
| --- | --- | --- | --- | --- |
| CCCC | C | C | C | C |
| CFFF | C | F | F | F |
| FCFC | C | C | F | F |
| CFCF | F | F | C | C |
Free edge (F): fix nothing (natural BC, same as PINN).
"t" below = derivative along the edge (y on x0/xa, x on y0/yb), "n" = normal derivative.

### Model A, clamp_type = full (Abaqus ENCASTRE; default; matches PINN v0.3)
- w_b: fix all 4 DOFs [w_b, w_b,x, w_b,y, w_b,xy] at every edge node (w_b = 0 and ∇w_b = 0 → both rotations 0).
- w_s: fix w_s and w_s,t only (w_s = 0 along the edge; w_s,n and w_s,xy stay free).

### Model A, clamp_type = senjanovic (w = 0, normal rotation 0) — do after "full" works
- w_b,n = 0 and w_b,xy = 0 (fix).
- w = 0 → w_s = −w_b and w_s,t = −w_b,t on the edge: implement with a transformation matrix T (slave DOF = −master DOF), K' = TᵀKT, M' = TᵀMT.
- **Trap:** the pair (w_b = c, w_s = −c) has zero K and zero M energy → singular. Remove it: additionally fix w_b (value) at ONE node of a clamped edge.

### Model B
- C: w = ψx = ψy = 0.
- (test only) S, hard simple support: w = 0 and tangential rotation ψ_t = 0.
- Model A test-only S: w_b = w_b,t = w_s = w_s,t = 0.

Apply BCs by deleting fixed DOFs (keep a `free` index array), not by big-number penalty.

---

## 6. Solve
- `scipy.sparse.linalg.eigsh(K_ff, k=n_modes+n_extra, M=M_ff, sigma=0.0, which='LM')` (shift-invert → lowest modes, fast).
- Sort ascending; ω = √eig; f = ω/2π; λ = ω² m a⁴ / D.
- Check: no negative or near-zero eigenvalues (would mean a missing BC or a null mode). Stop with a clear error if any.
- Mode shapes: eigsh gives Mᵀ-normalised vectors (φᵀMφ = 1). For plots: W normalised to max |W| = 1, sign so that the largest value is positive.
- Total deflection for plots and MAC: Model A W = w_b + w_s; Model B W = w.

---

## 7. Outputs (same style as PINN)
`results/<case>_<boundary>_fea_<model>/`
- `frequencies.json`: case, boundary, model, clamp_type, mesh, n_dof, solve_seconds, constants, and per mode: mode, f_Hz, lambda, omega_rad_s, mn_estimate, order_ok, ritz_Hz + error_vs_ritz_pct (Model A only), pinn_Hz + error_vs_pinn_pct (if PINN results exist).
- `modes_grid.npz`: W on the PINN post-processing grid (101 × 51), plus w_b, w_s (Model A) or ψx, ψy (Model B); x, y arrays.
- `mesh_convergence.csv`: mesh, n_dof, f1..f5, % change vs previous mesh.
- `orthogonality.csv`: φᵢᵀ M φⱼ (should be identity).
- `config.yaml` copy.
- Plots (3_postprocess_fea.py): contour + 3D of W per mode, same colour map and label "W (normalised)" as the PINN plots, frequency table PNG.

4_compare_all.py:
- Table per BC and mode: PINN | FEA-A | Ritz | FEA-B | Abaqus, and % differences.
- MAC matrix PINN vs FEA-A (should be ≈ 1 on the diagonal, this also closes the open "residual" item of the PINN report), FEA-A vs FEA-B, FEA vs Abaqus (reuse the Abaqus reader of `4_compare_abaqus.py`).
- MAC(φ, ψ) = (φᵀψ)² / ((φᵀφ)(ψᵀψ)) on the same grid points.

---

## 8. Tests (write and pass these BEFORE running the 4 BCs)
### 8.1 Exact Mindlin SSSS (both models)
For hard simply supported plate, mode (p, q): k² = (pπ/a)² + (qπ/b)², c_r = 1 + 2/((1−ν)k).
Solve (mJ/S) ω⁴ − (m + J c_r k²) ω² + D k⁴ = 0, take the SMALLER root.
case_05 values (Hz): 13.8789 (1,1), 21.0851 (2,1), 31.8078 (3,1), 39.5067 (1,2), 44.8779 (2,2) = 44.8779 (4,1) (double).
Pass: Model A 40 × 20 within 0.01 %; Model B 160 × 80 within 0.2 %.

### 8.2 Thin-plate limit
Set h very small (s = a/h = 1000): Model A and B must approach Kirchhoff: square CCCC λ₁ ≈ 35.99² (ω a² √(m/D) = 35.99, Leissa); SSSS λ = π⁴(1 + r²)². Model B must NOT lock (frequency must not blow up as h → 0).

### 8.3 Patch / sanity
- Free-free plate (no BC): exactly 3 zero modes (1 translation + 2 rotations), no extra zero modes (no spurious modes). Use sigma = −1.0 in eigsh for this test.
  - Model A trap: (w_b = 1, w_s = −1) has zero K AND zero M (w = 0, no rotation) → singular. For the free-free test only, fix w_s value at one node. Clamped cases do not have this problem (w_s = 0 on the clamped edge).
- Mass check: rigid translation vector t (all w value DOFs = 1, others 0; Model A: w_b value DOFs = 1) → tᵀ M t = ρ a b h = 0.52 kg.

---

## 9. Validation targets (case_05: 0.20 × 0.10 × 0.02 m)
These numbers come from a quick prototype of exactly this spec (Nov 24, 2026). Your code should reproduce them.

### 9.1 Model A (senjanovic, clamp full) — target = PINN/Ritz
| BC | mesh | f1 | f2 | f3 | f4 | f5 |
| --- | --- | --- | --- | --- | --- | --- |
| CCCC | 40×20 | 22.8954 | 29.2618 | 39.0385 | 49.5406 | 51.1656 |
| CCCC | 80×40 | 22.8954 | 29.2618 | 39.0384 | 49.5403 | 51.1653 |
| CFFF | 80×40 | 1.0124 | 3.8877 | 5.9013 | 12.2402 | 15.3614 |
| FCFC | 80×40 | 6.0917 | 9.4757 | 15.6467 | 20.3317 | 27.1739 |
| CFCF | 80×40 | 20.4818 | 21.8206 | 26.1187 | 33.2963 | 43.4857 |
| Ritz (N=11), CCCC | – | 22.895 | 29.262 | 39.038 | 49.540 | 51.165 |
| PINN v0.3, CCCC | – | 22.8967 | 29.2649 | 39.0409 | 49.5429 | 51.1693 |

- Frequencies fall monotonically with mesh refinement (BFS is conforming → upper bound).
- For free-edge BCs, converged FE is ~0.05–0.1 % BELOW Ritz N=11 → Ritz is not fully converged there; FEA-A becomes the better reference for the PINN.
- Pass rule: FEA-A within 0.15 % of PINN for all 20 modes, same order, MAC(PINN, FEA-A) > 0.99.
- clamp_type senjanovic CCCC target (Ritz): 22.375, 27.945, 37.300, 47.031, 49.259.

### 9.2 Model B (mindlin3, MITC4, clamp full)
| BC | mesh | f1 | f2 | f3 | f4 | f5 |
| --- | --- | --- | --- | --- | --- | --- |
| CCCC | 160×80 | 22.473 | 28.194 | 37.599 | 47.161 | 49.553 |
| CCCC reference (3-var Mindlin, spec_rq.md) | – | 22.468 | 28.188 | 37.588 | 47.133 | 49.530 |
| CFFF | 160×80 | 0.994 | 3.555 | 5.844 | 11.295 | 15.171 |
| FCFC | 160×80 | 6.006 | 8.477 | 15.092 | 18.505 | 26.759 |
| CFCF | 160×80 | 20.307 | 20.747 | 23.792 | 30.393 | 40.776 |
(MITC4 still converging slowly at 160×80; values are ~0.02–0.06 % high.)

### 9.3 Important finding to keep in the report (do not hide)
- Model B (standard Mindlin) is 2 % lower than Model A for CCCC, but **up to 9–11 % lower for some free-edge modes** (CFFF mode 2: 3.555 vs 3.888 Hz; FCFC mode 2: 8.477 vs 9.476 Hz; CFCF mode 3: 23.79 vs 26.12 Hz). These look like twisting-type modes.
- Cause: Model A forces ψ = ∇w_b (no "rotational" part of the rotation field). That part matters most near free edges and in twisting → Model A (and so the PINN) is stiffer there.
- This is a **theory** difference, not a code error. It can explain part of the PINN–Abaqus gap (report section 9.2). Confirm with: (1) MITC4 mesh convergence, (2) mode-shape plots (check those modes are twisting), (3) optional Abaqus S4R shell run, which is 3-field Mindlin like Model B.
- Abaqus 3D (C3D8RH) values are still indicative only (mesh not converged).

---

## 10. Order of work
1. Make folder, copy config, add `fea:` section. Read PINN files listed in section 1.
2. `hermite.py` + `element_senjanovic.py` + `test_patch.py` (free-free → 3 zero modes).
3. Assembly + BCs + solve → `test_ssss_exact.py` passes for Model A.
4. Model A CCCC clamp full, mesh 40×20 → compare with table 9.1. Then mesh study.
5. Model A for CFFF, FCFC, CFCF (only edges change).
6. `evaluate.py`, `3_postprocess_fea.py`, MAC vs PINN (`4_compare_all.py`).
7. Model B (MITC4) → SSSS test, thin-limit (no locking) test, then all 4 BCs, mesh study to 160×80.
8. clamp_type senjanovic for Model A (transformation matrix + pin one w_b DOF).
9. `run_all_fea.py`: 4 BCs × 2 models × meshes → `logs/fea_summary.csv`.
10. k-sensitivity (shear_factor 5/6, 0.822, 0.882), Model A and B, CCCC and CFFF (report next step item).

## 11. Coding rules
- Plain comments for a beginner: each function says what it computes and which equation (section number of this spec) it uses.
- All settings from config.yaml / command line; no hard-coded geometry or material.
- float64; vectorised NumPy; sparse matrices; print n_dof and solve time.
- Print a final table like the PINN summary: mode, f_FEA, (m,n), f_PINN, f_Ritz, error %.
- Never overwrite PINN results; FEA results go in their own `_fea_<model>` folders.
- At the end, list every new file and what it does.

## 12. Later (not now)
- Crack: FE handles a through-crack by splitting nodes along the crack line (two node sets, no connection). Model B is easier for this (C0 element). Can give reference frequencies vs crack length for the crack PINN and the inverse problem.
- Pre-stress / NLGEOM: add geometric stiffness K_σ from a static pre-load — only if the Abaqus Step-1 is shown to matter.
