"""
Этап 5 (вариант А): Монте-Карло переноса электронов вместо прямолинейной модели.

Модель одиночного рассеяния Джоя (D.C. Joy, «Monte Carlo Modeling for Electron
Microscopy and Microanalysis», 1995):
  • упругое рассеяние — экранированный Резерфорд с релятивистской поправкой;
  • потери энергии — формула Бете с модификацией Джоя–Луо (корректна при малых E);
  • каждое столкновение разыгрывается отдельно, траектории трёхмерные.
Применимость: ~0,5–100 кэВ, т.е. весь спектр Ni-63 (Emax 66,9 кэВ).

Что делаем:
  1. Проверка: коэффициент отражения η_bs(Z) против эмпирической формулы Ройтера.
  2. Выход энергии из слоя Ni-63 (с подложкой и без) против моделей этапов 2–4.
  3. Отражение от коллектора при реальных энергиях посадки.
  4. Пересчёт полного КПД прямого преобразования для Ni-63.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from direct_conversion import beta_spectrum, collector

rng = np.random.default_rng(7)
NA = 6.022e23
E_CUT = 0.5                                        # кэВ, ниже — электрон поглощён

MAT = {"C": dict(Z=6, A=12.01, rho=2.26), "Be": dict(Z=4, A=9.01, rho=1.85),
       "Al": dict(Z=13, A=26.98, rho=2.70), "Ni": dict(Z=28, A=63.0, rho=8.9),
       "Cu": dict(Z=29, A=63.55, rho=8.96), "Au": dict(Z=79, A=196.97, rho=19.3)}
for m in MAT.values():
    m["J"] = (9.76*m["Z"] + 58.5*m["Z"]**-0.19)*1e-3    # кэВ, средний потенциал ионизации
    m["N"] = NA*m["rho"]/m["A"]                          # атомов/см³

def reuter(Z):
    return -0.0254 + 0.016*Z - 1.86e-4*Z**2 + 8.3e-7*Z**3

def transport(E, z, w, layers):
    """Перенос пачки электронов через плоские слои [(z0, z1, материал), ...] (мкм),
    вне слоёв — вакуум. E — кэВ, z — мкм, w — (n,3) единичные направления.
    Возвращает энергию, cos угла к оси z и сторону выхода (-1 / +1) вышедших электронов."""
    zb = np.array([l[0] for l in layers] + [layers[-1][1]])
    mats = [MAT[l[2]] for l in layers]
    Zs = np.array([m["Z"] for m in mats]); As = np.array([m["A"] for m in mats])
    rhos = np.array([m["rho"] for m in mats]); Js = np.array([m["J"] for m in mats])
    Ns = np.array([m["N"] for m in mats])
    E, z, w = E.copy(), z.copy(), w.copy()
    outE, outc, outs = [], [], []
    alive = np.arange(len(E))
    for _ in range(200000):
        if alive.size == 0:
            break
        e, zz, ww = E[alive], z[alive], w[alive]
        k = np.clip(np.searchsorted(zb, zz, side="right") - 1, 0, len(layers) - 1)
        Z, A, rho, J, N = Zs[k], As[k], rhos[k], Js[k], Ns[k]
        a = 3.4e-3*Z**0.67/e                                            # экранирование
        sig = 5.21e-21*Z**2/e**2*4*np.pi/(a*(1 + a))*((e + 511)/(e + 1024))**2
        lam = 1e4/(N*sig)                                               # мкм
        s = -lam*np.log(rng.random(e.size))
        # не пересекаем границу слоя за один шаг
        lo, hi = zb[k], zb[k + 1]
        dz = ww[:, 2]
        to_b = np.where(dz > 0, (hi - zz)/np.maximum(dz, 1e-12),
                        np.where(dz < 0, (lo - zz)/np.minimum(dz, -1e-12), np.inf))
        cross = to_b < s
        s = np.where(cross, to_b + 1e-7, s)
        # потери энергии (Бете, Джой–Луо), кэВ/мкм
        dEds = 7.85e4*rho*Z/(A*e)*np.log(1.166*(e + 0.85*J)/J)*1e-4
        e = e - dEds*s
        zz = zz + ww[:, 2]*s
        # упругое рассеяние (только если шаг закончился внутри слоя)
        sc = ~cross
        if sc.any():
            R = rng.random(sc.sum()); aa = a[sc]
            ct = 1 - 2*aa*R/(1 + aa - R); st = np.sqrt(np.clip(1 - ct**2, 0, 1))
            ph = 2*np.pi*rng.random(sc.sum())
            u = ww[sc]
            # поворот направления на (θ, φ)
            ref = np.where(np.abs(u[:, 2:3]) < 0.9, [[0, 0, 1]], [[1, 0, 0]])
            p1 = np.cross(u, ref); p1 /= np.linalg.norm(p1, axis=1, keepdims=True)
            p2 = np.cross(u, p1)
            ww[sc] = (u*ct[:, None] + (p1*np.cos(ph)[:, None] + p2*np.sin(ph)[:, None])*st[:, None])
        E[alive], z[alive], w[alive] = e, zz, ww
        out = (zz < zb[0]) | (zz > zb[-1])
        dead = (e < E_CUT) & ~out
        if out.any():
            outE.append(e[out]); outc.append(np.abs(ww[out, 2])); outs.append(np.sign(zz[out] - zb[0]))
        alive = alive[~(out | dead)]
    cat = lambda x: np.concatenate(x) if x else np.array([])
    return cat(outE), cat(outc), cat(outs)

def iso_dirs(n):
    c = rng.uniform(-1, 1, n); p = 2*np.pi*rng.random(n); s = np.sqrt(1 - c**2)
    return np.stack([s*np.cos(p), s*np.sin(p), c], axis=1)

def backscatter(mat, E0, n=4000):
    """Нормальное падение на толстую мишень: доля отражённых по числу и по энергии."""
    R = 50.0
    w = np.tile([0.0, 0.0, 1.0], (n, 1))
    E, c, side = transport(np.full(n, float(E0)), np.full(n, 1e-6), w, [(0, R, mat)])
    b = side < 0
    return b.sum()/n, E[b].sum()/(n*E0)

def source_escape(t, n=20000, substrate=None):
    """Слой Ni-63 толщиной t мкм (опционально на подложке (материал, толщина)).
    Возвращает долю энергии распада, вышедшую наружу, и вышедшие энергии и cos."""
    E0 = beta_spectrum(66.9, 29, n)
    z = rng.random(n)*t
    layers = [(0.0, t, "Ni")]
    if substrate:
        layers.append((t, t + substrate[1], substrate[0]))
    E, c, _ = transport(E0, z, iso_dirs(n), layers)
    return E.sum()/E0.sum(), E, c, E0.sum()

if __name__ == "__main__":
    # --- 1. Проверка модели по отражению
    print("=== 1. Проверка: коэффициент отражения η_bs (нормальное падение, 20 кэВ)")
    print(f"{'мат.':5}{'Z':>4}{'MC':>8}{'Ройтер':>9}   доля энергии (MC)")
    val = {}
    for m in ["Be", "C", "Al", "Cu", "Au"]:
        nb, eb = backscatter(m, 20)
        val[m] = nb
        print(f"{m:5}{MAT[m]['Z']:4}{nb:8.3f}{reuter(MAT[m]['Z']):9.3f}   {eb:.3f}")

    # --- 2. Выход энергии из слоя Ni-63
    mu = 17*0.0669**-1.14*8.9*1e-4                     # 1/мкм, модель этапов 2–3
    print("\n=== 2. Выход энергии из свободного слоя Ni-63 (обе грани)")
    print(f"{'t,мкм':>7}{'MC':>8}{'экспон.(эт.2-3)':>17}")
    ts = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    esc_mc = []
    for t in ts:
        f, *_ = source_escape(t, n=8000)
        esc_mc.append(f)
        print(f"{t:7.2f}{f*100:7.1f}%{(1 - np.exp(-mu*t))/(mu*t)*100:15.1f}%")

    print("\n=== 2б. Слой 0,2 мкм на подложке (электроны сквозь подложку тоже считаются)")
    for sub in [None, ("C", 0.1), ("C", 1.0), ("Ni", 1.0), ("Au", 1.0)]:
        f, *_ = source_escape(0.2, n=8000, substrate=sub)
        print(f"  подложка {str(sub):14}: выход {f*100:5.1f}%")

    # --- 3. Отражение от коллектора при энергии посадки
    print("\n=== 3. Отражение от коллектора (доля по числу) от энергии посадки")
    Eland = np.array([1, 2, 5, 10, 20, 40])
    bs_tab = {m: np.array([backscatter(m, e, n=2500)[0] for e in Eland]) for m in ["Be", "C", "Cu"]}
    for m, v in bs_tab.items():
        print(f"  {m:3}: " + "  ".join(f"{e} кэВ: {x:.3f}" for e, x in zip(Eland, v)))

    # --- 4. Полный КПД для Ni-63, сферическая схема
    print("\n=== 4. Ni-63, слой 0,2 мкм, сферическая схема: пересчёт КПД")
    f, E, c, Etot = source_escape(0.2, n=30000)
    q = np.ones_like(E)
    for Nst in [1, 5, 20]:
        got, V = collector(E, q, Nst)
        V = np.sort(V)
        # энергия посадки: электрон с энергией E садится на самый высокий V_k ≤ E
        k = np.searchsorted(V, E, side="right") - 1
        land = E - V[np.clip(k, 0, None)]
        ok = k >= 0
        loss = {}
        for m in ["Be", "C", "Cu"]:
            p_bs = np.interp(land[ok], Eland, bs_tab[m])
            loss[m] = (p_bs*V[k[ok]]).sum()/got     # отражённый электрон уносит заряд обратно
        eta = got/Etot
        print(f"  N={Nst:2}: выход×коллектор = {eta*100:5.1f}%; потери на отражение: "
              + ", ".join(f"{m} {loss[m]*100:4.1f}%" for m in loss)
              + f"; итог (C, утечки 0,99, кВ→В 0,90) = {eta*(1 - loss['C'])*0.99*0.90*100:4.1f}%")

    # --- графики
    fig, ax = plt.subplots(3, 1, figsize=(7, 13))
    Zs = np.linspace(3, 82, 100)
    ax[0].plot(Zs, reuter(Zs), label="Ройтер (эмпирика)")
    ax[0].scatter([MAT[m]["Z"] for m in val], list(val.values()), c="r", zorder=5, label="Монте-Карло, 20 кэВ")
    for m in val:
        ax[0].annotate(m, (MAT[m]["Z"], val[m]), xytext=(4, -10), textcoords="offset points")
    ax[0].set_xlabel("Атомный номер Z"); ax[0].set_ylabel("Коэффициент отражения")
    ax[0].set_title("Проверка модели"); ax[0].grid(alpha=.3); ax[0].legend()

    tt = np.geomspace(0.02, 5, 50)
    ax[1].plot(ts, np.array(esc_mc)*100, "o-", label="Монте-Карло (рассеяние)")
    ax[1].plot(tt, (1 - np.exp(-mu*tt))/(mu*tt)*100, "--", label="Экспоненциальная (этапы 2–3)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Толщина слоя Ni-63, мкм")
    ax[1].set_ylabel("Вышедшая энергия, % от распада"); ax[1].set_title("Самопоглощение Ni-63")
    ax[1].grid(alpha=.3); ax[1].legend()

    ax[2].hist(beta_spectrum(66.9, 29, 30000), bins=60, range=(0, 67), density=True, histtype="step", lw=2, label="спектр распада")
    ax[2].hist(E, bins=60, range=(0, 67), density=True, histtype="step", lw=2, label="вышедшие из слоя 0,2 мкм")
    ax[2].set_xlabel("Энергия электрона, кэВ"); ax[2].set_ylabel("Плотность")
    ax[2].set_title("Спектр Ni-63 до и после выхода"); ax[2].grid(alpha=.3); ax[2].legend()
    plt.tight_layout(); plt.savefig("mc_electrons.png", dpi=130)
