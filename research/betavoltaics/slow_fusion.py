"""
Этап 7: энергобаланс «медленного» синтеза — есть ли шанс на плюс?

Считаем коэффициент усиления Q = (энергия синтеза) / (затраченная энергия) для
  1. пучка ионов в холодную мишень (генераторы нейтронов, ускорители);
  2. пучка ионов в горячую плазму (торможение по Спитцеру);
  3. мюонного катализа (синтез при комнатной температуре).
И сравниваем с порогом окупаемости при прямом сборе заряженных продуктов полем.

Сечения: Bosch & Hale, Nucl. Fusion 32 (1992) 611 (D-T, D-D); D-He3 — аппроксимация
Дуэйна (NRL Plasma Formulary); p-B11 — Nevins & Swain, Nucl. Fusion 40 (2000) 865 ниже
400 кэВ, выше — резонанс 612 кэВ с фоном 0,15 б (грубо, ±50%).
Торможение в холодной мишени: Андерсен–Циглер для водорода, масштаб по числу электронов
(±30%). Все упрощения — в пользу синтеза: вывод «не окупается» от них устойчив.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def bosch_hale(E, BG, A, B):                         # E — кэВ (СЦМ), σ — барн
    S = (A[0] + E*(A[1] + E*(A[2] + E*(A[3] + E*A[4]))))/(1 + E*(B[0] + E*(B[1] + E*(B[2] + E*B[3]))))
    return S/(E*np.exp(BG/np.sqrt(E)))*1e-3

def sigma_DT(E):
    return bosch_hale(E, 34.3827, [6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0], [63.8, -0.995, 6.981e-5, 1.728e-4])
def sigma_DD(E):                                     # обе ветви
    return (bosch_hale(E, 31.3970, [5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9], [0, 0, 0, 0]) +
            bosch_hale(E, 31.3970, [5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10], [0, 0, 0, 0]))
def sigma_DHe3(E):                                   # аппроксимация Дуэйна (NRL Formulary), E_лаб дейтрона
    El = E*5/3
    return (647 + 25900/((1.297 - 3.98e-3*El)**2 + 1))/(El*(np.exp(89.27/np.sqrt(El)) - 1))
def sigma_pB11(E):
    S = 197 + 0.24*E + 2.31e-4*E**2 + 1.82e4/((E - 148)**2 + 2.35**2)   # МэВ·барн, Невинс–Суэйн (< 400 кэВ)
    low = S*1e3/(E*np.exp(np.sqrt(22589/E)))
    high = 0.15 + 0.9/(1 + ((E - 612)/150)**2)       # резонанс 612 кэВ (СЦМ), сшито при 400 кэВ, ±50%
    return np.where(E < 400, low, high)

# Реакции: пучок, мишень, массы (а.е.м.), Z, сечение, энергия (МэВ), доля в заряженных частицах,
# мишень: электронов на ядро мишени; плазма: Σ n_j Z_j² /(n_e A_j), n_t/n_e
R = {
    "D→T":    dict(Ab=2, At=3,  Zb=1, sig=sigma_DT,   Ef=17.59, fch=0.20, ze=1, zsum=1/3,  nt=1.0),
    "D→D":    dict(Ab=2, At=2,  Zb=1, sig=sigma_DD,   Ef=3.65,  fch=0.66, ze=1, zsum=1/2,  nt=1.0),
    "D→He3":  dict(Ab=2, At=3,  Zb=1, sig=sigma_DHe3, Ef=18.35, fch=0.95, ze=2, zsum=2/3,  nt=0.5),
    "p→B11":  dict(Ab=1, At=11, Zb=1, sig=sigma_pB11, Ef=8.68,  fch=1.00, ze=5, zsum=5/11, nt=0.2),
}

def S_cold(E_lab, Ab, ze):                           # эВ·см² на ядро мишени
    e = E_lab/Ab                                     # кэВ/а.е.м.
    lo = 1.44*e**0.45
    hi = 242.6/e*np.log(1 + 1.2e4/e + 0.1159*e)
    return ze*lo*hi/(lo + hi)*1e-15

def Q_cold(r, E0):
    E = np.linspace(1, E0, 3000)
    Ecm = E*r["At"]/(r["Ab"] + r["At"])
    Y = np.trapezoid(r["sig"](Ecm)*1e-24/S_cold(E, r["Ab"], r["ze"]), E*1e3)   # вероятность синтеза
    return Y, Y*r["Ef"]*1e3/E0

def Q_plasma(r, E0, Te_keV, lnL=17.0):
    """Пучок тормозится на электронах и ионах плазмы (Спитцер); выход не зависит от плотности."""
    E = np.linspace(max(1.0, 1.5*Te_keV), E0, 3000)
    if E0 <= E[0]:
        return 0.0, 0.0
    Ec = 14.8*Te_keV*r["Ab"]*r["zsum"]**(2/3)
    tau_ne = 6.27e8*r["Ab"]*(Te_keV*1e3)**1.5/(r["Zb"]**2*lnL)          # τ_s·n_e, с/см³
    v = np.sqrt(2*E*1.602e-16/(r["Ab"]*1.6605e-27))*100                # см/с
    Ecm = E*r["At"]/(r["Ab"] + r["At"])
    dEdt_over_ne = 2*E/tau_ne*(1 + (Ec/E)**1.5)                         # кэВ·см³/с
    Y = np.trapezoid(r["nt"]*r["sig"](Ecm)*1e-24*v/dEdt_over_ne, E)
    return Y, Y*r["Ef"]*1e3/E0

if __name__ == "__main__":
    ETA_DC, ETA_ACC = 0.70, 0.90                      # сбор полем (этапы 4–5), ускоритель
    print("=== Порог окупаемости по электричеству при сборе заряженных продуктов полем")
    print(f"    Q_нужно = 1/(f_заряж·η_поля·η_ускор),  η_поля = {ETA_DC}, η_ускор = {ETA_ACC}")
    for n, r in R.items():
        print(f"  {n:7} f_заряж = {r['fch']:.2f} → Q > {1/(r['fch']*ETA_DC*ETA_ACC):5.1f}")

    Es = np.geomspace(20, 5000, 120)
    print("\n=== 1. Пучок в холодную мишень (лучшая энергия пучка)")
    cold = {}
    for n, r in R.items():
        q = np.array([Q_cold(r, e)[1] for e in Es]); cold[n] = q
        i = q.argmax(); Y = Q_cold(r, Es[i])[0]
        print(f"  {n:7} Q_max = {q[i]:.2e} при {Es[i]:6.0f} кэВ; вероятность синтеза {Y:.1e} на ион")
    y150 = Q_cold(R["D→T"], 150)[0]
    print(f"  проверка: D→T при 150 кэВ, выход {y150:.1e} на дейтрон (генераторы нейтронов: ~1e-5–1e-4)")

    print("\n=== 2. Пучок в горячую плазму (оптимальная энергия пучка для каждой Te)")
    Tes = np.array([0.1, 1, 3, 10, 30, 100, 300])
    hot = {}
    for n, r in R.items():
        best = []
        for Te in Tes:
            q = [Q_plasma(r, e, Te)[1] for e in Es]; best.append(max(q))
        hot[n] = np.array(best)
        print(f"  {n:7} " + "  ".join(f"Te={Te:g} кэВ: {b:.2g}" for Te, b in zip(Tes, best)))

    print("\n=== 3. Мюонный катализ (d-t, комнатная температура — единственный «холодный» синтез)")
    tau_mu = 2.197e-6
    print("  X = 1/(1/(λc·τμ) + ωs) — число синтезов на мюон")
    E_el = 14.1*0.35 + 3.5*ETA_DC                    # МэВ электричества на синтез: нейтрон → тепло 35%, α → поле
    print(f"  электричество на синтез: нейтрон 14,1 МэВ × 0,35 + α 3,5 МэВ × {ETA_DC} = {E_el:.1f} МэВ")
    for lc, ws, tag in [(1.2e8, 0.0045, "эксперимент (λc ≈ 1,2·10⁸/с, ωs ≈ 0,45%)"),
                        (1.2e8, 0.0010, "если снизить прилипание до 0,1%"),
                        (5e8, 0.0010, "и ускорить цикл в 4 раза")]:
        X = 1/(1/(lc*tau_mu) + ws)
        print(f"  {tag}: X ≈ {X:.0f}")
        for Emu in [3, 5, 8]:
            print(f"      цена мюона {Emu} ГэВ: Q_эл = {X*E_el/(Emu*1e3):.2f}")
    print(f"  для Q_эл = 1 при цене 5 ГэВ нужно X ≈ {5e3/E_el:.0f} синтезов на мюон")

    # --- графики
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 14))
    Ecm = np.geomspace(3, 3000, 400)
    for n, r in R.items():
        ax[0].loglog(Ecm, r["sig"](Ecm), label=n.replace("→", "+"))
    ax[0].set_ylim(1e-6, 10); ax[0].set_xlabel("Энергия в системе центра масс, кэВ")
    ax[0].set_ylabel("Сечение, барн"); ax[0].set_title("Сечения реакций синтеза"); ax[0].grid(alpha=.3, which="both")
    ax[0].legend(fontsize=8)

    for n, q in cold.items():
        ax[1].loglog(Es, q, label=n)
    for n, r in R.items():
        ax[1].axhline(1/(r["fch"]*ETA_DC*ETA_ACC), ls=":", lw=0.8, c="gray")
    ax[1].text(25, 2.2, "пороги окупаемости (1,6–8)", fontsize=8)
    ax[1].set_ylim(1e-7, 30); ax[1].set_xlabel("Энергия пучка, кэВ"); ax[1].set_ylabel("Q = E_синтеза / E_пучка")
    ax[1].set_title("Пучок в холодную мишень"); ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)

    for n, q in hot.items():
        ax[2].loglog(Tes, np.maximum(q, 1e-9), "o-", label=n)
    ax[2].axhspan(1.6, 8, color="gray", alpha=0.15); ax[2].text(0.12, 2.3, "зона окупаемости", fontsize=8)
    ax[2].set_ylim(1e-6, 100); ax[2].set_xlabel("Температура электронов плазмы-мишени, кэВ")
    ax[2].set_ylabel("Лучший Q"); ax[2].set_title("Пучок в горячую плазму (без затрат на её поддержание)")
    ax[2].grid(alpha=.3, which="both"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("slow_fusion.png", dpi=130)
