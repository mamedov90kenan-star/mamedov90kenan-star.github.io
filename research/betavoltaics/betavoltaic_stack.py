"""
Этап 3: многослойная структура [источник Ni-63 | диод | источник | диод ...]
Периодическая стопка, 1D-модель экспоненциального ослабления бета-потока,
перенос электронов через слои суммируется геометрическим рядом.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

kT, Q, CI = 0.02585, 1.602e-19, 3.7e10
MU_MASS = 17 * 0.0669**-1.14                      # см²/г, Ni-63 (Emax 66.9 кэВ)
SA, RHO_NI, EMEAN = 56.7, 8.9, 17.4e3             # Ки/г, г/см³, эВ
P_VOL = SA * RHO_NI * CI * EMEAN * Q              # Вт/см³ распада (чистый Ni-63)
T_HALF = 100.1

MAT = {"Алмаз": dict(Eg=5.47, rho=3.52), "4H-SiC": dict(Eg=3.26, rho=3.21)}
mu_s = MU_MASS * RHO_NI * 1e-4                    # 1/мкм

def eps(Eg): return 2.8*Eg + 0.5

def electric(P_col, Eg, J0_factor):
    Jsc = P_col / eps(Eg)
    J0 = 1.5e5*np.exp(-Eg/kT)*J0_factor
    Voc = kT*np.log(Jsc/J0 + 1); v = Voc/kT
    FF = (v - np.log(v+0.72))/(v+1)
    return Jsc*Voc*FF

def stack(ts, td, mat, Lc, J0_factor=1.0):
    """ts, td в мкм. Возвращает КПД системы и мощность на объём."""
    mu_d = MU_MASS*MAT[mat]["rho"]*1e-4
    Ts, Td = np.exp(-mu_s*ts), np.exp(-mu_d*td)
    f_esc = (1 - Ts)/(mu_s*ts)                    # доля энергии, вышедшая из слоя источника
    # доля входящего в диод потока, поглощённая в зоне сбора (Lc от каждой грани)
    c = np.where(td <= 2*Lc, 1 - Td,
                 (1 - np.exp(-mu_d*Lc)) + (np.exp(-mu_d*(td-Lc)) - Td))
    F = c / (1 - Td*Ts)                           # с учётом многократного прохода по стопке
    P_src = P_VOL*ts*1e-4                         # Вт/см² распада в одном слое
    P_col = P_src*f_esc*F                         # собранная мощность на один диод
    P_el = electric(P_col, MAT[mat]["Eg"], J0_factor)
    eta = P_el/P_src
    P_v = P_el/((ts+td)*1e-4)                     # Вт/см³ стопки
    return eta, P_v, f_esc, F

ts = np.geomspace(0.05, 10, 160)
td = np.geomspace(0.5, 80, 160)
TS, TD = np.meshgrid(ts, td)

scen = [("Алмаз", 20, 1), ("Алмаз", 5, 1), ("Алмаз", 20, 1e10), ("4H-SiC", 20, 1), ("4H-SiC", 5, 1e10)]
res = {}
print(f"Мощность распада чистого Ni-63: {P_VOL*1e3:.1f} мВт/см³, {P_VOL/RHO_NI*1e3:.2f} мВт/г\n")
for mat, Lc, jf in scen:
    eta, Pv, fe, F = stack(TS, TD, mat, Lc, jf)
    res[(mat, Lc, jf)] = (eta, Pv)
    i = np.unravel_index(np.argmax(eta), eta.shape)
    j = np.unravel_index(np.argmax(Pv), Pv.shape)
    tag = f"{mat}, Lc={Lc} мкм, J0×{jf:.0e}"
    print(f"=== {tag}")
    print(f"  Макс. КПД:     η={eta[i]*100:5.1f}%  ts={TS[i]:.2f} td={TD[i]:.1f} мкм  P={Pv[i]*1e3:.2f} мВт/см³")
    print(f"  Макс. мощность: η={eta[j]*100:5.1f}%  ts={TS[j]:.2f} td={TD[j]:.1f} мкм  P={Pv[j]*1e3:.2f} мВт/см³"
          f"  слоёв на мм: {1000/(TS[j]+TD[j]):.0f}")

# Компромиссная точка: макс. мощность при η ≥ 90% от максимума
eta, Pv = res[("Алмаз", 20, 1e10)]
mask = eta >= 0.9*eta.max()
k = np.unravel_index(np.argmax(np.where(mask, Pv, 0)), Pv.shape)
ts_k, td_k, P_k, e_k = TS[k], TD[k], Pv[k], eta[k]
avg = (1-2**(-100/T_HALF))/(np.log(2)/T_HALF)     # эффективные годы работы за 100 лет
E100 = P_k*avg*8766                               # Вт·ч/см³
E50 = P_k*(1-2**(-50/T_HALF))/(np.log(2)/T_HALF)*8766
print(f"\n=== Компромисс (реалистичный алмаз, Lc=20): ts={ts_k:.2f} td={td_k:.1f} мкм, η={e_k*100:.1f}%, P={P_k*1e3:.2f} мВт/см³")
print(f"  Мощность через 50 лет: {P_k*0.5**(50/T_HALF)*1e3:.2f} мВт/см³, через 100 лет: {P_k*0.5**(100/T_HALF)*1e3:.2f}")
print(f"  Энергия за 50 лет: {E50:.0f} Вт·ч/л, за 100 лет: {E100:.0f} Вт·ч/л  (Li-ion ≈ 700 Вт·ч/л, один цикл)")
print(f"  Для 1 мВт нужно {1e-3/P_k:.2f} см³ стопки и {1e-3/(e_k*P_VOL/RHO_NI):.2f} г Ni-63")

# Влияние обогащения (Ni-63 в смеси со стабильным Ni): КПД тот же, мощность ∝ обогащению
for enr in [0.15, 0.5, 1.0]:
    print(f"  Обогащение {enr*100:.0f}%: {P_k*enr*1e3:.2f} мВт/см³")

# --- графики
fig, ax = plt.subplots(3, 1, figsize=(7, 14))
eta, Pv = res[("Алмаз", 20, 1e10)]
cs = ax[0].contourf(TS, TD, eta*100, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax[0], label="КПД системы, %")
c2 = ax[0].contour(TS, TD, Pv*1e3, levels=[0.5,1,2,4,6], colors="w", linewidths=0.8)
ax[0].clabel(c2, fmt="%g мВт/см³", fontsize=7)
ax[0].scatter(ts_k, td_k, c="r", s=60, zorder=5, label="компромисс")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("Толщина слоя Ni-63, мкм"); ax[0].set_ylabel("Толщина диода, мкм")
ax[0].set_title("Алмаз, Lc=20 мкм, реалистичный J0: КПД и мощность"); ax[0].legend()

for (mat, Lc, jf), (e, p) in res.items():
    order = np.argsort(p.ravel())
    pp, ee = p.ravel()[order], e.ravel()[order]
    front = np.maximum.accumulate(ee[::-1])[::-1]   # Парето: макс. η при мощности ≥ P
    ax[1].plot(pp*1e3, front*100, label=f"{mat}, Lc={Lc}, J0×{jf:.0e}")
ax[1].set_xlabel("Мощность, мВт/см³"); ax[1].set_ylabel("Макс. КПД системы, %")
ax[1].set_title("Фронт Парето: КПД против мощности"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)

for Lc in [2, 5, 20, 100]:
    e, _, _, _ = stack(ts, np.full_like(ts, 2*Lc if Lc < 40 else 60), "Алмаз", Lc, 1e10)
    ax[2].plot(ts, e*100, label=f"Lc={Lc} мкм")
ax[2].set_xscale("log"); ax[2].set_xlabel("Толщина слоя Ni-63, мкм"); ax[2].set_ylabel("КПД системы, %")
ax[2].set_title("Алмаз: влияние длины сбора носителей"); ax[2].grid(alpha=.3); ax[2].legend()
plt.tight_layout(); plt.savefig("betavoltaic_stack.png", dpi=130)
