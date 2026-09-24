"""
Модель: техническое задание на идеальный бетавольтаический материал.
Этап 2 исследования. Все допущения помечены в комментариях.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

kT = 0.02585            # эВ, 300 K
Q = 1.602e-19
CI = 3.7e10             # Бк на кюри
MEC2 = 0.511            # МэВ
AMU = 931.494           # МэВ

# --- Изотопы: Emax, Emean (кэВ), T1/2 (лет), удельная активность (Ки/г, чистый изотоп)
ISO = {
    "H-3":   dict(Emax=18.6,  Emean=5.7,  T=12.32, SA=9619),
    "Ni-63": dict(Emax=66.9,  Emean=17.4, T=100.1, SA=56.7),
    "Pm-147":dict(Emax=224.6, Emean=61.9, T=2.62,  SA=928),
}

# --- Материалы: Eg (эВ), Z, A, плотность, самый лёгкий атом (а.е.м.), порог смещения Ed (эВ, нижняя оценка)
MAT = {
    "Si":      dict(Eg=1.12, Z=14, A=28.09, rho=2.33, M=28.09, Ed=15),
    "GaAs":    dict(Eg=1.42, Z=32, A=72.3,  rho=5.32, M=69.72, Ed=10),
    "4H-SiC":  dict(Eg=3.26, Z=10, A=20.05, rho=3.21, M=12.01, Ed=20),
    "GaN":     dict(Eg=3.39, Z=19, A=41.87, rho=6.15, M=14.01, Ed=20),
    "Алмаз":   dict(Eg=5.47, Z=6,  A=12.01, rho=3.52, M=12.01, Ed=37),
}

def eps_pair(Eg):            # энергия на пару, формула Клейна
    return 2.8 * Eg + 0.5

def J0(Eg):                  # идеальный ток насыщения диода (эмпирика Грина), А/см²
    return 1.5e5 * np.exp(-Eg / kT)

def converter(Eg, P_in, CE=1.0, bs=0.0):
    """КПД преобразования P_in (Вт/см²) в электричество."""
    P_abs = P_in * (1 - bs)
    Jsc = P_abs * CE / eps_pair(Eg)                 # А/см²
    Voc = kT * np.log(Jsc / J0(Eg) + 1)
    voc = Voc / kT
    FF = (voc - np.log(voc + 0.72)) / (voc + 1)
    P_el = Jsc * Voc * FF
    return P_el / P_in, Jsc, Voc, FF, P_el

def range_um(E_keV, m):      # пробег электрона, Канайя–Окаяма
    return 0.0276 * m["A"] * E_keV**1.67 / (m["Z"]**0.889 * m["rho"])

def backscatter(Z):          # доля обратного рассеяния (Ройтер)
    return -0.0254 + 0.016*Z - 1.86e-4*Z**2 + 8.3e-7*Z**3

def Tmax_eV(E_keV, M_amu):   # макс. энергия, переданная атому электроном
    E = E_keV / 1000
    return 2*E*(E + 2*MEC2) / (M_amu*AMU) * 1e6

# --- Источник Ni-63: выход мощности с поверхности от толщины
ni = ISO["Ni-63"]
mu = 17 * (ni["Emax"]/1000)**-1.14 * 8.9          # 1/см, эмпирика ослабления бета
t_um = np.linspace(0.05, 15, 300)
P_decay_vol = ni["SA"] * 8.9 * CI * ni["Emean"]*1e3 * Q   # Вт/см³
P_surf = 0.5 * P_decay_vol * (1 - np.exp(-mu * t_um*1e-4)) / mu   # Вт/см², в сторону диода
P_sat = 0.5 * P_decay_vol / mu
t95 = -np.log(0.05) / mu * 1e4

print("=== ИСТОЧНИК Ni-63 (чистый изотоп) ===")
print(f"1/μ = {1/mu*1e4:.2f} мкм; 95% насыщения при t = {t95:.1f} мкм")
print(f"Макс. поток мощности к диоду: {P_sat*1e6:.1f} мкВт/см²")
print(f"Слой толщиной t95 использует {P_sat*0.95/(P_decay_vol*t95*1e-4)*100:.0f}% энергии распада\n")

P_in = P_sat * 0.95

print("=== МАТЕРИАЛЫ под Ni-63 (идеальный диод) ===")
print(f"{'Мат.':8}{'ε,эВ':>6}{'Eg/ε':>7}{'обр.расс':>9}{'Voc,В':>7}{'FF':>6}{'η_конв':>8}{'R_ср,мкм':>9}{'R_max,мкм':>10}")
for n, m in MAT.items():
    bs = backscatter(m["Z"]) * 0.6        # энергетическая доля ~0.6 от числовой
    eta, Jsc, Voc, FF, _ = converter(m["Eg"], P_in, 1.0, bs)
    print(f"{n:8}{eps_pair(m['Eg']):6.2f}{m['Eg']/eps_pair(m['Eg']):7.2f}{bs*100:8.1f}%"
          f"{Voc:7.2f}{FF:6.2f}{eta*100:7.1f}%{range_um(ni['Emean'],m):9.2f}{range_um(ni['Emax'],m):10.1f}")

print("\n=== РАДИАЦИОННАЯ СТОЙКОСТЬ: T_max (эВ) против порога Ed ===")
print(f"{'Мат.':8}{'Ed':>5}" + "".join(f"{k:>10}" for k in ISO))
for n, m in MAT.items():
    row = f"{n:8}{m['Ed']:5}"
    for k, iso in ISO.items():
        T = Tmax_eV(iso["Emax"], m["M"])
        row += f"{T:7.1f}{' ok' if T < m['Ed'] else ' !!'}"
    print(row)
Eth = {}
for n, m in MAT.items():   # пороговая энергия электрона, выше которой начинается повреждение
    E = np.linspace(1, 2000, 20000)
    Eth[n] = E[np.argmax(Tmax_eV(E, m["M"]) >= m["Ed"])]
print("Порог повреждения (кэВ):", {k: round(v) for k, v in Eth.items()})

# --- Непрерывная развёртка по Eg
Eg = np.linspace(0.5, 7, 400)
eta_ideal = np.array([converter(e, P_in)[0] for e in Eg])
eta_ideal_sun = Eg/eps_pair(Eg)

# --- Чувствительность к длине сбора (W+L) для Ni-63
def CE_of(WL, R):
    x0 = R / 2.5
    return 1 - np.exp(-WL / x0)

print("\n=== Требуемая длина сбора W+L для CE≥95% (Ni-63, по R_max) ===")
for n, m in MAT.items():
    R = range_um(ni["Emax"], m)
    print(f"{n:8} W+L ≥ {-np.log(0.05)*R/2.5:5.1f} мкм")

# --- Чувствительность к качеству материала: J0 хуже идеального в 10^k раз
print("\n=== Потеря КПД при дефектах (J0 × 10^k), 4H-SiC и алмаз ===")
for n in ["4H-SiC", "Алмаз"]:
    m = MAT[n]; out = []
    for k in [0, 10, 20, 30]:
        P_abs = P_in*(1-backscatter(m["Z"])*0.6)
        Jsc = P_abs/eps_pair(m["Eg"])
        Voc = kT*np.log(Jsc/(J0(m["Eg"])*10**k)+1); voc=Voc/kT
        FF=(voc-np.log(voc+0.72))/(voc+1)
        out.append(f"10^{k}: {Jsc*Voc*FF/P_in*100:4.1f}%")
    print(n, " | ".join(out))

# --- Графики
fig, ax = plt.subplots(3, 1, figsize=(7, 13))
ax[0].plot(Eg, eta_ideal*100, lw=2, label="Идеальный диод (Voc·FF учтены)")
ax[0].plot(Eg, eta_ideal_sun*100, "--", label="Предел Eg/ε (без потерь напряжения)")
for n, m in MAT.items():
    e = converter(m["Eg"], P_in)[0]*100
    ax[0].scatter(m["Eg"], e, zorder=5); ax[0].annotate(n, (m["Eg"], e), xytext=(4,-12), textcoords="offset points")
ax[0].set_xlabel("Ширина запрещённой зоны Eg, эВ"); ax[0].set_ylabel("КПД преобразователя, %")
ax[0].set_title("Теоретический КПД от Eg (Ni-63)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

ax[1].plot(t_um, P_surf*1e6, lw=2)
ax[1].axvline(t95, ls="--", c="gray"); ax[1].text(t95, P_sat*0.5e6, f" t95 = {t95:.1f} мкм")
ax[1].set_xlabel("Толщина слоя Ni-63, мкм"); ax[1].set_ylabel("Мощность к диоду, мкВт/см²")
ax[1].set_title("Самопоглощение источника"); ax[1].grid(alpha=.3)

E = np.linspace(1, 400, 400)
for n, m in MAT.items():
    ax[2].plot(E, Tmax_eV(E, m["M"])/m["Ed"], label=n)
ax[2].axhline(1, c="k", lw=1)
for k, iso in ISO.items():
    ax[2].axvline(iso["Emax"], ls=":", c="gray"); ax[2].text(iso["Emax"], 3.3, k, rotation=90, fontsize=8)
ax[2].set_ylim(0, 3.5); ax[2].set_xlabel("Энергия электрона, кэВ"); ax[2].set_ylabel("T_max / Ed  (>1 = повреждение)")
ax[2].set_title("Радиационная стойкость"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("betavoltaic_spec.png", dpi=130)
