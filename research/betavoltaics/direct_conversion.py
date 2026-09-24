"""
Этап 4: общая теория «вещество + поле» для всех видов радиации.

Цепочка преобразования (одна для всех видов излучения):
  η = η_захв · η_вых · η_геом · η_колл(N) · η_доп
  η_захв  — доля энергии излучения, переданная заряженным частицам в веществе
            (для α, β и продуктов реакций = 1; для гамма — поглощение в фольгах)
  η_вых   — доля кинетической энергии, с которой частицы выходят из слоя в вакуум
  η_геом  — доля энергии, направленная вдоль тормозящего поля
            (плоская схема: E·cos²θ; магнитное «сопло»; сферическая схема ≈ 1)
  η_колл  — многоступенчатый электростатический коллектор с N электродами
  η_доп   — отражённые/вторичные частицы, утечки, преобразователь кВ→В

Модель — оценочная: прямолинейные траектории, степенной закон пробега
(E_ост = E0·(1 − s/R)^(1/p)), изотропное рождение частиц. Для публикации
нужен Монте-Карло (Geant4 / PENELOPE). Константы помечены как приближённые.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
ME, ALPHA_FS = 511.0, 1/137.036                  # кэВ; постоянная тонкой структуры
Q, CI = 1.602e-19, 3.7e10
ETA_AUX = 0.97 * 0.99 * 0.90                      # отражение/вторичные · утечки · кВ→В
NS = 60000                                        # число частиц Монте-Карло

# ---------------------------------------------------------------- спектры
def beta_spectrum(Qkev, Zd, n):
    """Разрешённый бета-спектр с нерелятивистской функцией Ферми."""
    E = np.linspace(0.05, Qkev, 4000)
    W = E + ME; p = np.sqrt(W**2 - ME**2); b = p/W
    x = 2*np.pi*Zd*ALPHA_FS/b
    N = p*W*(Qkev - E)**2 * x/(1 - np.exp(-x))
    c = np.cumsum(N); c /= c[-1]
    return np.interp(rng.random(n), c, E)

def compton_photo(Eg, f_pe, Ebind, n):
    """Электроны, выбитые гамма-квантом: фотоэффект + Клейн–Нишина."""
    k = Eg/ME; Tmax = Eg*2*k/(1 + 2*k)
    T = np.linspace(1e-3, Tmax*0.9999, 4000); s = T/Eg
    d = 2 + s**2/(k**2*(1 - s)**2) + s/(1 - s)*(s - 2/k)
    c = np.cumsum(d); c /= c[-1]
    e = np.interp(rng.random(n), c, T)
    pe = rng.random(n) < f_pe
    e[pe] = Eg - Ebind
    return e

# ------------------------------------------------ выход из слоя (Монте-Карло)
def escape(E0, R0, p, q, t):
    """Частицы рождаются равномерно по толщине t, изотропно; выходят через обе грани.
    E0 — энергии (кэВ), R0 — пробеги (мкм), p — показатель закона пробег–энергия.
    Возвращает энергии, cos угла к нормали и заряды вышедших частиц."""
    n = len(E0)
    z = rng.random(n)*t
    mu = rng.uniform(-1, 1, n)
    s = np.where(mu > 0, (t - z)/np.maximum(mu, 1e-9), z/np.maximum(-mu, 1e-9))
    ok = s < R0
    E = E0[ok]*(1 - s[ok]/R0[ok])**(1/p)
    return E, np.abs(mu[ok]), q[ok]

# ------------------------------------------ многоступенчатый коллектор (ДП)
def collector(Eu, qq, N):
    """Eu — энергия вдоль поля (кэВ), qq — заряд частиц (в e). N электродов с
    оптимальными напряжениями: частица доходит до самого высокого электрода с
    V ≤ Eu/q и отдаёт q·V. Возвращает собранную энергию (кэВ) и напряжения (кВ)."""
    Eq = Eu/qq
    o = np.argsort(Eq); Eq, qs = Eq[o], qq[o]
    G = np.quantile(Eq, np.linspace(0, 0.999, 300))
    cq = np.concatenate([np.cumsum(qs[::-1])[::-1], [0]])   # заряд частиц с Eq ≥ V
    f = cq[np.searchsorted(Eq, G)]
    m = len(G)
    dp = np.full((N + 1, m), -np.inf); arg = np.zeros((N + 1, m), int)
    dp[1] = G*f
    for k in range(2, N + 1):
        for j in range(m - 1):
            v = G[j]*(f[j] - f[j + 1:]) + dp[k - 1, j + 1:]
            a = np.argmax(v); dp[k, j] = v[a]; arg[k, j] = j + 1 + a
    j = int(np.argmax(dp[N])); best = dp[N, j]; V = [G[j]]
    for k in range(N, 1, -1):
        j = arg[k, j]; V.append(G[j])
    return best, np.array(V)

def geometry(E, mu, mode, RB=10.0):
    """Энергия, пригодная для торможения полем."""
    if mode == "плоская":
        return E*mu**2
    if mode == "магн. сопло":                    # E⊥/B сохраняется: E⊥ → E⊥/RB
        return E - E*(1 - mu**2)/RB
    return E                                       # сферическая, источник мал

# ---------------------------------------------------------- виды излучения
def ko_range(E, A, Z, rho):                       # электроны, Канайя–Окаяма, мкм
    return 0.0276*A*E**1.67/(Z**0.889*rho)

def bk_alpha_range(E_mev, sqrtA, rho):            # альфа, Брэгг–Климан, мкм
    return 3.2e-4*sqrtA/rho*0.318*E_mev**1.5*1e4

SQA_PUO2 = 1/(0.881/np.sqrt(238) + 0.119/4.0)

def src_Ni63(n):
    E = beta_spectrum(66.9, 29, n); return E, ko_range(E, 63, 28, 8.9), 1.67, 1
def src_Sr90(n):                                  # Sr-90 + Y-90 в равновесии, SrTiO3
    E = np.concatenate([beta_spectrum(546, 39, n//2), beta_spectrum(2280, 40, n - n//2)])
    return E, ko_range(E, 36.7, 16.8, 5.1), 1.67, 1
def src_Pu238(n):                                 # PuO2, α 5,5 МэВ, заряд 2e
    E = np.full(n, 5500.0); return E, np.full(n, bk_alpha_range(5.5, SQA_PUO2, 11.5)), 1.5, 2
def src_B10(n):                                   # 10B(n,α)7Li: α 1,47 МэВ + Li 0,84 МэВ
    a = rng.random(n) < 0.5                       # (пробеги в боре ~3,6 и ~1,9 мкм)
    E = np.where(a, 1470.0, 840.0); R = np.where(a, 3.6, 1.9)
    return E, R, 1.5, np.where(a, 2, 3)
def src_UO2(n):                                   # осколки деления, UO2, заряд ~ +20e
    L = rng.random(n) < 0.5                       # пробеги ~9 и ~7 мкм (грубо)
    E = np.where(L, 95e3, 67e3); R = np.where(L, 9.0, 7.0)
    return E, R, 0.6, 20

# гамма: μ/ρ, μ_en/ρ (см²/г, свинец, ±15% — сверить с NIST XCOM), доля фотоэффекта
GAMMA = {"γ Cs-137 (662 кэВ)": dict(E=662, mu=0.110, muen=0.071, fpe=0.45),
         "γ Co-60 (1,25 МэВ)":  dict(E=1250, mu=0.059, muen=0.035, fpe=0.20)}
RHO_PB = 11.35
def src_gamma(g):
    def f(n):
        E = compton_photo(g["E"], g["fpe"], 88.0, n)
        return E, ko_range(E, 207.2, 82, RHO_PB), 1.67, 1
    return f

SOURCES = {
    "β Ni-63":           (src_Ni63,  "Ni, распад"),
    "β Sr-90/Y-90":      (src_Sr90,  "SrTiO3, распад"),
    "α Pu-238":          (src_Pu238, "PuO2, распад"),
    "n → B-10 (α+Li)":   (src_B10,   "нейтроны, конвертер"),
    "n → U-235 (деление)": (src_UO2, "нейтроны, конвертер"),
    **{k: (src_gamma(v), "Pb-фольги") for k, v in GAMMA.items()},
}

def analyze(fn, t, modes=("плоская", "магн. сопло", "сферическая"), stages=(1, 5, 20)):
    E0, R0, p, q = fn(NS)
    q = np.broadcast_to(q, E0.shape).astype(float)
    E, mu, qq = escape(E0, R0, p, q, t)
    total = E0.sum()
    out = {"esc": E.sum()/total}
    for md in modes:
        Eu = geometry(E, mu, md)
        for N in stages:
            got, V = collector(Eu, qq, N)
            out[(md, N)], out[("V", md, N)] = got/total, V
    return out

# ------------------------------------------------------------------- расчёт
if __name__ == "__main__":
    thick = {"β Ni-63": 0.2, "β Sr-90/Y-90": 20, "α Pu-238": 1.0, "n → B-10 (α+Li)": 0.3,
             "n → U-235 (деление)": 0.5, "γ Cs-137 (662 кэВ)": 10, "γ Co-60 (1,25 МэВ)": 20}
    print("КПД цепочки для слоя/фольги указанной толщины (без η_захв для гамма)\n")
    hdr = f"{'излучение':22}{'t,мкм':>7}{'выход':>7}" + "".join(
        f"{m[:5]+' N='+str(N):>13}" for m in ("плоская", "магн. сопло", "сферическая") for N in (1, 20))
    print(hdr)
    res = {}
    for name, (fn, _) in SOURCES.items():
        r = analyze(fn, thick[name], stages=(1, 20)); res[name] = r
        row = f"{name:22}{thick[name]:7.1f}{r['esc']*100:6.0f}%"
        for m in ("плоская", "магн. сопло", "сферическая"):
            for N in (1, 20):
                row += f"{r[(m, N)]*100:12.0f}%"
        print(row)
    print(f"\nИтог с η_доп = {ETA_AUX:.2f} (сферическая, 20 ступеней):")
    for name, r in res.items():
        print(f"  {name:22} η ≈ {r[('сферическая', 20)]*ETA_AUX*100:4.0f}%"
              f"   напряжение верхней ступени ≈ {r[('V', 'сферическая', 20)].max()/1000:.3g} МВ")

    # --- Гамма: сколько фольг и какой длины нужна стопка
    print("\nГамма: стопка Pb-фольг с вакуумными зазорами (поглощение 90% энергии)")
    for name, g in GAMMA.items():
        d = thick[name]
        T = -np.log(0.1)/(g["muen"]*RHO_PB)*1e4                 # мкм свинца суммарно
        Nf = T/d
        V1 = res[name][("V", "плоская", 1)].max()                   # кВ, одна ступень
        gap = V1*1e3/5e6*1e3                                     # мм при 5 МВ/м
        eta = 0.9*res[name][("плоская", 1)]*ETA_AUX
        print(f"  {name}: свинца {T/1e4:.1f} см → {Nf:.0f} фольг по {d} мкм, "
              f"зазор ≥ {gap:.1f} мм → длина ≈ {Nf*(gap + d/1e3)/1e3:.1f} м; η ≈ {eta*100:.1f}%")

    # --- графики
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 14))
    for name, (fn, _) in SOURCES.items():
        tt = np.geomspace(0.02, 200, 30)
        ax[0].plot(tt, [analyze(fn, t, modes=("сферическая",), stages=(5,))[("сферическая", 5)]*100
                        for t in tt], label=name)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Толщина слоя / фольги, мкм")
    ax[0].set_ylabel("η_вых·η_колл, %  (сфер., 5 ступеней)")
    ax[0].set_title("Самопоглощение: чем тоньше слой, тем выше КПД"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)

    Ns = [1, 2, 3, 5, 10, 20]
    for name, (fn, _) in SOURCES.items():
        r = analyze(fn, thick[name], modes=("сферическая",), stages=Ns)
        ax[1].plot(Ns, [r[("сферическая", N)]*ETA_AUX*100 for N in Ns], "o-", label=name)
    ax[1].axhline(28, ls="--", c="gray"); ax[1].text(12, 29, "алмазный диод (Ni-63), 28%", fontsize=8)
    ax[1].set_xlabel("Число ступеней коллектора"); ax[1].set_ylabel("Полный КПД, %")
    ax[1].set_title("Прямое преобразование полем (сферическая схема)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)

    names = list(SOURCES); x = np.arange(len(names)); w = 0.27
    for i, m in enumerate(("плоская", "магн. сопло", "сферическая")):
        ax[2].bar(x + (i - 1)*w, [res[n][(m, 20)]*ETA_AUX*100 for n in names], w, label=m)
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax[2].set_ylabel("Полный КПД, % (20 ступеней)"); ax[2].set_title("Роль геометрии поля")
    ax[2].grid(alpha=.3, axis="y"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("direct_conversion.png", dpi=130)
