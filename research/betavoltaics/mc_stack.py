"""
Этап 5б (А2): алмазная стопка [Ni-63 | диод | Ni-63 | диод ...] методом Монте-Карло.

Та же модель Джоя, что в mc_electrons.py, но геометрия бесконечно-периодическая:
координата z сворачивается по периоду ts+td, электрон движется, пока не остановится.
Считается, куда уходит энергия распада: в никель (самопоглощение), в зону сбора
диода (±Lc от граней) или в «мёртвую» середину толстого диода. Отражение от
алмаза обратно в никель, блуждание и многократные проходы учтены автоматически.
Электрическая часть (Voc, FF, J0) — как в betavoltaic_stack.py (этап 3).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from direct_conversion import beta_spectrum
from mc_electrons import MAT, E_CUT

rng = np.random.default_rng(11)
kT, Q, CI = 0.02585, 1.602e-19, 3.7e10
P_VOL = 56.7*8.9*CI*17.4e3*Q                        # Вт/см³ распада, чистый Ni-63
EG = {"Алмаз": 5.47, "4H-SiC": 3.26}
NB = 400                                             # бины глубины в диоде

def electric(P_col, Eg, J0_factor):                  # как в этапе 3
    Jsc = P_col/(2.8*Eg + 0.5)
    J0 = 1.5e5*np.exp(-Eg/kT)*J0_factor
    Voc = kT*np.log(Jsc/J0 + 1); v = Voc/kT
    return Jsc*Voc*(v - np.log(v + 0.72))/(v + 1)

def periodic_stack(ts, td, diode="Алмаз", n=3000):
    """Доли энергии распада: в никеле и гистограмма по глубине в диоде (NB бинов)."""
    P = ts + td
    mats = [MAT["Ni"], MAT[diode]]
    Zs = np.array([m["Z"] for m in mats]); As = np.array([m["A"] for m in mats])
    rhos = np.array([m["rho"] for m in mats]); Js = np.array([m["J"] for m in mats])
    Ns = np.array([m["N"] for m in mats])
    E0 = beta_spectrum(66.9, 29, n)
    E = E0.copy(); z = rng.random(n)*ts
    c = rng.uniform(-1, 1, n); ph = 2*np.pi*rng.random(n); s_ = np.sqrt(1 - c**2)
    w = np.stack([s_*np.cos(ph), s_*np.sin(ph), c], axis=1)
    dep_ni, hist = 0.0, np.zeros(NB)

    def deposit(zpos, dE):
        nonlocal dep_ni
        u = np.mod(zpos, P)
        inNi = u < ts
        dep_ni += dE[inNi].sum()
        d = u[~inNi] - ts
        hist[:] += np.bincount(np.clip((d/td*NB).astype(int), 0, NB - 1), dE[~inNi], NB)

    while E.size:
        u = np.mod(z, P); k = (u >= ts).astype(int)
        Z, A, rho, J, N = Zs[k], As[k], rhos[k], Js[k], Ns[k]
        a = 3.4e-3*Z**0.67/E
        sig = 5.21e-21*Z**2/E**2*4*np.pi/(a*(1 + a))*((E + 511)/(E + 1024))**2
        s = -1e4/(N*sig)*np.log(rng.random(E.size))
        base = z - u
        lo = base + np.where(k == 0, 0.0, ts); hi = base + np.where(k == 0, ts, P)
        dz = w[:, 2]
        to_b = np.where(dz > 0, (hi - z)/np.maximum(dz, 1e-12),
                        np.where(dz < 0, (lo - z)/np.minimum(dz, -1e-12), np.inf))
        cross = to_b < s
        s = np.where(cross, to_b + 1e-7, s)
        dE = np.minimum(7.85e4*rho*Z/(A*E)*np.log(1.166*(E + 0.85*J)/J)*1e-4*s, E)
        zmid = z + dz*s*0.5
        deposit(zmid, dE)
        E = E - dE; z = z + dz*s
        sc = ~cross
        if sc.any():
            R = rng.random(sc.sum()); aa = a[sc]
            ct = 1 - 2*aa*R/(1 + aa - R); st = np.sqrt(np.clip(1 - ct**2, 0, 1))
            f = 2*np.pi*rng.random(sc.sum()); uu = w[sc]
            ref = np.where(np.abs(uu[:, 2:3]) < 0.9, [[0, 0, 1]], [[1, 0, 0]])
            p1 = np.cross(uu, ref); p1 /= np.linalg.norm(p1, axis=1, keepdims=True)
            p2 = np.cross(uu, p1)
            w[sc] = uu*ct[:, None] + (p1*np.cos(f)[:, None] + p2*np.sin(f)[:, None])*st[:, None]
        stop = E < E_CUT
        if stop.any():
            deposit(z[stop], E[stop])
        keep = ~stop
        E, z, w = E[keep], z[keep], w[keep]
    tot = E0.sum()
    return dep_ni/tot, hist/tot

def collected(hist, td, Lc):
    d = (np.arange(NB) + 0.5)/NB*td
    return hist[np.minimum(d, td - d) < Lc].sum()

if __name__ == "__main__":
    TS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])
    TD = np.array([0.3, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 40])
    ni = np.zeros((len(TD), len(TS))); H = {}
    for i, td in enumerate(TD):
        for j, ts in enumerate(TS):
            ni[i, j], H[(i, j)] = periodic_stack(ts, td)
    print(f"Прогнано {len(TS)*len(TD)} конфигураций стопки (по 3000 электронов)\n")

    # аналитика этапа 3 для сравнения (доля энергии, собранной в диоде)
    MU = 17*0.0669**-1.14*1e-4
    def analytic(ts, td, Lc):
        ms, md = MU*8.9, MU*3.52
        Ts, Td = np.exp(-ms*ts), np.exp(-md*td)
        fe = (1 - Ts)/(ms*ts)
        c = 1 - Td if td <= 2*Lc else (1 - np.exp(-md*Lc)) + (np.exp(-md*(td - Lc)) - Td)
        return fe*c/(1 - Td*Ts)

    print("Доля энергии распада, собранная в алмазе (Lc = 20 мкм): МК / этап 3")
    print("td\\ts " + "".join(f"{t:>11}" for t in TS[[1, 4, 6, 8]]))
    for i in [1, 2, 4, 6, 9]:
        print(f"{TD[i]:5}" + "".join(f"{collected(H[(i, j)], TD[i], 20)*100:6.0f}/{analytic(TS[j], TD[i], 20)*100:3.0f}%"
                                     for j in [1, 4, 6, 8]))

    def scan(Lc, jf, diode="Алмаз"):
        eta = np.zeros_like(ni); Pv = np.zeros_like(ni)
        for i, td in enumerate(TD):
            for j, ts in enumerate(TS):
                P_src = P_VOL*ts*1e-4
                P_el = electric(P_src*collected(H[(i, j)], td, Lc), EG[diode], jf)
                eta[i, j] = P_el/P_src; Pv[i, j] = P_el/((ts + td)*1e-4)
        return eta, Pv

    print()
    for Lc, jf, tag in [(20, 1, "идеальный алмаз"), (20, 1e10, "реалистичный алмаз (J0×1e10)"), (5, 1e10, "плохой алмаз (Lc=5, J0×1e10)")]:
        eta, Pv = scan(Lc, jf)
        a = np.unravel_index(eta.argmax(), eta.shape); b = np.unravel_index(Pv.argmax(), Pv.shape)
        m = eta >= 0.9*eta.max(); c = np.unravel_index(np.where(m, Pv, 0).argmax(), Pv.shape)
        print(f"=== {tag}, Lc={Lc} мкм")
        for nm, p in [("макс. КПД", a), ("макс. мощность", b), ("компромисс", c)]:
            print(f"  {nm:15} η={eta[p]*100:5.1f}%  ts={TS[p[1]]:.2f} td={TD[p[0]]:.1f} мкм  P={Pv[p]*1e3:.2f} мВт/см³")
        if jf == 1e10 and Lc == 20:
            best = (eta, Pv, c)

    # --- технологическое ограничение: алмазный диод ≥ 1 мкм, слой Ni-63 ≥ 0,05 мкм
    eta, Pv, _ = best
    ok = (TD[:, None] >= 1) & (TS[None, :] >= 0.05)
    a = np.unravel_index(np.where(ok, eta, 0).argmax(), eta.shape)
    b = np.unravel_index(np.where(ok, Pv, 0).argmax(), Pv.shape)
    m = ok & (eta >= 0.9*eta[a]); c = np.unravel_index(np.where(m, Pv, 0).argmax(), Pv.shape)
    print("=== реалистичный алмаз с ограничением td ≥ 1 мкм, ts ≥ 0,05 мкм")
    for nm, p in [("макс. КПД", a), ("макс. мощность", b), ("компромисс", c)]:
        print(f"  {nm:15} η={eta[p]*100:5.1f}%  ts={TS[p[1]]:.2f} td={TD[p[0]]:.1f} мкм  P={Pv[p]*1e3:.2f} мВт/см³")

    # --- предел тонких слоёв: стопка = однородная «смесь», важна только x = ts/td
    def S(m, E=15.0):                                # тормозная способность (Бете), отн. ед.
        mm = MAT[m]; return mm["rho"]*mm["Z"]/mm["A"]*np.log(1.166*(E + 0.85*mm["J"])/mm["J"])
    kS = S("Ni")/S("Алмаз")
    print(f"\nПредел тонких слоёв: kS = S_Ni/S_алмаз = {kS:.2f}")
    print(f"  η(x) = η_conv/(1 + {kS:.2f}·x),  P(x) = P_распада·η_conv·x/((1+x)(1+{kS:.2f}·x))")
    xo = 1/np.sqrt(kS)
    print(f"  макс. мощность при x = 1/√kS = {xo:.2f}: доля энергии в алмазе {1/(1+kS*xo)*100:.0f}%,"
          f" P ≈ {P_VOL*0.245*xo/((1+xo)*(1+kS*xo))*1e3:.2f} мВт/см³ (η_conv≈24,5%)")
    x9 = (1/0.9 - 1)/kS
    print(f"  компромисс (90% от η_conv) при x = {x9:.3f}: P ≈ {P_VOL*0.245*0.9*x9/(1+x9)*1e3:.2f} мВт/см³")

    print("\nЭтап 3 (аналитика), реалистичный алмаз: макс. η 24,7%; макс. P 1,95 мВт/см³ при 9,7%;"
          " компромисс 22,3% / 0,55 мВт/см³ (ts 0,26, td 5,3)")

    # --- графики
    eta, Pv, c = best
    fig, ax = plt.subplots(2, 1, figsize=(7, 10))
    cs = ax[0].contourf(TS, TD, eta*100, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax[0], label="КПД системы, %")
    c2 = ax[0].contour(TS, TD, Pv*1e3, levels=[0.1, 0.25, 0.5, 1], colors="w", linewidths=0.8)
    ax[0].clabel(c2, fmt="%g мВт/см³", fontsize=7)
    ax[0].scatter(TS[c[1]], TD[c[0]], c="r", s=60, zorder=5, label="компромисс")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Толщина слоя Ni-63, мкм"); ax[0].set_ylabel("Толщина алмаза, мкм")
    ax[0].set_title("Монте-Карло: алмазная стопка, Lc=20 мкм, J0×1e10"); ax[0].legend()

    def front(e, p):
        o = np.argsort(p.ravel()); pp, ee = p.ravel()[o], e.ravel()[o]
        return pp, np.maximum.accumulate(ee[::-1])[::-1]
    pp, ff = front(eta, Pv)
    ax[1].plot(pp*1e3, ff*100, "o-", label="Монте-Карло (этап 5б)")
    ea = np.zeros_like(eta); pa = np.zeros_like(eta)
    for i, td in enumerate(TD):
        for j, ts in enumerate(TS):
            P_src = P_VOL*ts*1e-4; P_el = electric(P_src*analytic(ts, td, 20), 5.47, 1e10)
            ea[i, j] = P_el/P_src; pa[i, j] = P_el/((ts + td)*1e-4)
    pp, ff = front(ea, pa)
    ax[1].plot(pp*1e3, ff*100, "s--", label="Аналитика этапа 3 (та же сетка)")
    x = np.geomspace(1e-3, 1/np.sqrt(kS), 300)          # только ветвь Парето
    ax[1].plot(P_VOL*0.245*x/((1 + x)*(1 + kS*x))*1e3, 24.5/(1 + kS*x), ":", c="k",
               label=f"Предел тонких слоёв: η = η_conv/(1+{kS:.2f}·ts/td)")
    ax[1].set_xlabel("Мощность, мВт/см³"); ax[1].set_ylabel("Макс. КПД системы, %")
    ax[1].set_title("Фронт Парето: КПД против мощности"); ax[1].grid(alpha=.3); ax[1].legend()
    plt.tight_layout(); plt.savefig("mc_stack.png", dpi=130)
