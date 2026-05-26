# -*- coding: utf-8 -*-
import subprocess
import sys

try:
    from neuron import h, gui
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "neuron"])
    from neuron import h, gui

import random
import numpy as np
import matplotlib.pyplot as plt

h.load_file('stdrun.hoc')

# ========== ПРАВИЛЬНЫЙ КОНФИГ (~1340 нейронов) ==========
# В соответствии с YAML-файлом
N_thalamus_E = 80
N_thalamus_I = 20
N_L4_E = 200
N_L4_I = 40
N_L23_E = 150   # создаст 300 (RS + FRB)
N_L23_I = 40    # создаст 120 (Bask, LTS, Axax)
N_L5_E = 120    # создаст 240 (RS + IB)
N_L5_I = 40     # часть L56_I
N_L6_E = 130
N_L6_I = 30     # часть L56_I

h.tstop = 100.0
h.dt = 0.1

class HHNeuron:
    def __init__(self, inh=False):
        self.soma = h.Section(name='soma')
        self.soma.L = 20
        self.soma.diam = 20
        self.soma.insert('hh')
        self.inh = inh
        self.vvec = h.Vector()
        self.vvec.record(self.soma(0.5)._ref_v)
        self.tvec = h.Vector()
        self.tvec.record(h._ref_t)

def split_population(population, n_subgroups):
    size = len(population)
    step = size // n_subgroups
    return [population[i*step:(i+1)*step] for i in range(n_subgroups)]

def flatten(population):
    if population and isinstance(population[0], list):
        return [neuron for subgroup in population for neuron in subgroup]
    return population

def connect_exc(source_neurons, target_neurons, e=0, tau=2.0, threshold=0, weight=0.001, delay=3.0):
    source_neurons = flatten(source_neurons)
    target_neurons = flatten(target_neurons)
    netcons = []
    synapses = []
    for src in source_neurons:
        for tgt in target_neurons:
            syn = h.ExpSyn(tgt.soma(0.5))
            syn.e = e
            syn.tau = tau
            nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
            nc.threshold = threshold
            nc.weight[0] = weight
            nc.delay = delay
            synapses.append(syn)
            netcons.append(nc)
    return synapses, netcons

def connect_exc_gauss(source_neurons, target_neurons, e=0, tau=2.0, threshold=0, weight_mean=0.001, weight_std=0.0009, delay_mean=3.0, delay_std=2):
    source_neurons = flatten(source_neurons)
    target_neurons = flatten(target_neurons)
    netcons = []
    synapses = []
    for src in source_neurons:
        for tgt in target_neurons:
            syn = h.ExpSyn(tgt.soma(0.5))
            syn.e = e
            syn.tau = tau
            nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
            nc.threshold = threshold
            w = max(0.0, random.gauss(weight_mean, weight_std))
            d = max(0.1, random.gauss(delay_mean, delay_std))
            nc.weight[0] = w
            nc.delay = d
            synapses.append(syn)
            netcons.append(nc)
    return synapses, netcons

def connect_inh(source_neurons, target_neurons, e=-75, tau=3.0, threshold=0, weight=0.001, delay=2.0):
    source_neurons = flatten(source_neurons)
    target_neurons = flatten(target_neurons)
    netcons = []
    synapses = []
    for src in source_neurons:
        for tgt in target_neurons:
            syn = h.ExpSyn(tgt.soma(0.5))
            syn.e = e
            syn.tau = tau
            nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
            nc.threshold = threshold
            nc.weight[0] = weight
            nc.delay = delay
            synapses.append(syn)
            netcons.append(nc)
    return synapses, netcons

print("="*60)
print("СОЗДАНИЕ НЕЙРОНОВ (~1340 нейронов)")
print("="*60)

# Таламус
thalamus_E_TCR = [HHNeuron(inh=False) for _ in range(N_thalamus_E)]
thalamus_I_nRT = [HHNeuron(inh=True) for _ in range(N_thalamus_I)]
print(f"Таламус: {len(thalamus_E_TCR)} E + {len(thalamus_I_nRT)} I")

# L4
L4_E_Spinstel4 = split_population([HHNeuron() for _ in range(N_L4_E)], 1)
L4_I_LTS4 = split_population([HHNeuron(inh=True) for _ in range(N_L4_I)], 1)
print(f"L4: {len(flatten(L4_E_Spinstel4))} E + {len(flatten(L4_I_LTS4))} I")

# L2/3
L23_I_Axax23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], 3)
L23_I_Bask23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], 3)
L23_I_LTS23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], 3)
L23_E_SyppyrRS = split_population([HHNeuron() for _ in range(N_L23_E)], 2)
L23_E_SyppyrFRB = split_population([HHNeuron() for _ in range(N_L23_E)], 2)
print(f"L2/3: {len(flatten(L23_E_SyppyrRS)) + len(flatten(L23_E_SyppyrFRB))} E + {len(flatten(L23_I_Axax23))} I")

# L5
L5_E_TuftRS5 = split_population([HHNeuron() for _ in range(N_L5_E)], 2)
L5_E_TuftIB5 = split_population([HHNeuron() for _ in range(N_L5_E)], 2)

# L5/6 ингибиторы
total_I_56 = N_L5_I + N_L6_I
L56_I_Bask56 = split_population([HHNeuron(inh=True) for _ in range(total_I_56)], 3)
L56_I_Axax56 = split_population([HHNeuron(inh=True) for _ in range(total_I_56)], 3)
L56_I_LTS56 = split_population([HHNeuron(inh=True) for _ in range(total_I_56)], 3)
print(f"L5: {len(flatten(L5_E_TuftRS5)) + len(flatten(L5_E_TuftIB5))} E")

# L6
L6_E_NontuftRS6 = split_population([HHNeuron() for _ in range(N_L6_E)], 1)
print(f"L6: {len(flatten(L6_E_NontuftRS6))} E")
print(f"L5/6 I: {len(flatten(L56_I_Bask56))} ингибиторов")

# Подсчёт общего числа
total_neurons = (len(thalamus_E_TCR) + len(thalamus_I_nRT) +
                 len(flatten(L4_E_Spinstel4)) + len(flatten(L4_I_LTS4)) +
                 len(flatten(L23_E_SyppyrRS)) + len(flatten(L23_E_SyppyrFRB)) +
                 len(flatten(L23_I_Axax23)) + len(flatten(L23_I_Bask23)) + len(flatten(L23_I_LTS23)) +
                 len(flatten(L5_E_TuftRS5)) + len(flatten(L5_E_TuftIB5)) +
                 len(flatten(L6_E_NontuftRS6)) +
                 len(flatten(L56_I_Bask56)) + len(flatten(L56_I_Axax56)) + len(flatten(L56_I_LTS56)))
print(f"\n✅ ВСЕГО НЕЙРОНОВ: {total_neurons}")

print("\n" + "="*60)
print("ГЕНЕРАТОРЫ СПАЙКОВ ДЛЯ ТАЛАМУСА")
print("="*60)

# Генератор спайков для таламуса
syn_inputs = []
conns = []
netstims = []

for cell in thalamus_E_TCR:
    netstim = h.NetStim()
    netstim.start = 10
    netstim.number = 10
    netstim.interval = 20
    netstim.noise = 0.5
    syn = h.ExpSyn(cell.soma(0.5))
    syn.e = 0
    syn.tau = 2.0
    nc = h.NetCon(netstim, syn)
    nc.weight[0] = 0.01
    syn_inputs.append(syn)
    conns.append(nc)
    netstims.append(netstim)

print(f"Создано {len(netstims)} NetStim для таламических нейронов")

print("\n" + "="*60)
print("СОЗДАНИЕ СВЯЗЕЙ")
print("="*60)

# Таламус → L4
synapses_TCR_to_L4, netcons_TCR_to_L4 = connect_exc_gauss(thalamus_E_TCR, L4_E_Spinstel4)
print(f"Таламус → L4: {len(synapses_TCR_to_L4)} связей")

# Таламус → nRT (ингибиторы)
synapses_TCR_to_nRT, netcons_TCR_to_nRT = connect_exc(thalamus_E_TCR, thalamus_I_nRT)
print(f"Таламус → nRT: {len(synapses_TCR_to_nRT)} связей")

# L4 → L4 (рекуррентные)
synapses_L4_to_L4, netcons_L4_to_L4 = connect_exc_gauss(L4_E_Spinstel4, L4_E_Spinstel4)
print(f"L4 → L4: {len(synapses_L4_to_L4)} связей")

# L4 → L4 ингибиторы
synapses_L4_to_L4_I_LTS4, netcons_L4_to_L4_I_LTS4 = connect_exc(L4_E_Spinstel4, L4_I_LTS4)
print(f"L4 → L4 I: {len(synapses_L4_to_L4_I_LTS4)} связей")

# L4 → L2/3
synapses_L4_to_L23_RS, netcons_L4_to_L23_RS = connect_exc(L4_E_Spinstel4, L23_E_SyppyrRS)
synapses_L4_to_L23_FRB, netcons_L4_to_L23_FRB = connect_exc(L4_E_Spinstel4, L23_E_SyppyrFRB)
print(f"L4 → L2/3: {len(synapses_L4_to_L23_RS) + len(synapses_L4_to_L23_FRB)} связей")

# L2/3 → L2/3
synapses_L23_RS_to_RS, netcons_L23_RS_to_RS = connect_exc(L23_E_SyppyrRS, L23_E_SyppyrRS)
synapses_L23_FRB_to_FRB, netcons_L23_FRB_to_FRB = connect_exc(L23_E_SyppyrFRB, L23_E_SyppyrFRB)
print(f"L2/3 → L2/3: {len(synapses_L23_RS_to_RS) + len(synapses_L23_FRB_to_FRB)} связей")

# L2/3 → L5
synapses_L23_RS_to_TuftRS5, netcons_L23_RS_to_TuftRS5 = connect_exc(L23_E_SyppyrRS, L5_E_TuftRS5)
synapses_L23_RS_to_TuftIB5, netcons_L23_RS_to_TuftIB5 = connect_exc(L23_E_SyppyrRS, L5_E_TuftIB5)
synapses_L23_FRB_to_TuftRS5, netcons_L23_FRB_to_TuftRS5 = connect_exc(L23_E_SyppyrFRB, L5_E_TuftRS5)
synapses_L23_FRB_to_TuftIB5, netcons_L23_FRB_to_TuftIB5 = connect_exc(L23_E_SyppyrFRB, L5_E_TuftIB5)
print(f"L2/3 → L5: {len(synapses_L23_RS_to_TuftRS5) + len(synapses_L23_RS_to_TuftIB5) + len(synapses_L23_FRB_to_TuftRS5) + len(synapses_L23_FRB_to_TuftIB5)} связей")

# L5 → L5
synapses_TuftRS5_to_TuftRS5, netcons_TuftRS5_to_TuftRS5 = connect_exc(L5_E_TuftRS5, L5_E_TuftRS5)
synapses_TuftRS5_to_TuftIB5, netcons_TuftRS5_to_TuftIB5 = connect_exc(L5_E_TuftRS5, L5_E_TuftIB5)
synapses_TuftIB5_to_TuftIB5, netcons_TuftIB5_to_TuftIB5 = connect_exc(L5_E_TuftIB5, L5_E_TuftIB5)
print(f"L5 → L5: {len(synapses_TuftRS5_to_TuftRS5) + len(synapses_TuftRS5_to_TuftIB5) + len(synapses_TuftIB5_to_TuftIB5)} связей")

# L5 → L6
synapses_TuftRS5_to_L6, netcons_TuftRS5_to_L6 = connect_exc(L5_E_TuftRS5, L6_E_NontuftRS6)
synapses_TuftIB5_to_L6, netcons_TuftIB5_to_L6 = connect_exc(L5_E_TuftIB5, L6_E_NontuftRS6)
print(f"L5 → L6: {len(synapses_TuftRS5_to_L6) + len(synapses_TuftIB5_to_L6)} связей")

# L6 → L5 (обратные связи)
synapses_L6_to_TuftRS5, netcons_L6_to_TuftRS5 = connect_exc(L6_E_NontuftRS6, L5_E_TuftRS5)
synapses_L6_to_TuftIB5, netcons_L6_to_TuftIB5 = connect_exc(L6_E_NontuftRS6, L5_E_TuftIB5)
print(f"L6 → L5: {len(synapses_L6_to_TuftRS5) + len(synapses_L6_to_TuftIB5)} связей")

# Тормозные связи
synapses_LTS4_to_L4, netcons_LTS4_to_L4 = connect_inh(L4_I_LTS4, L4_E_Spinstel4)
print(f"L4 I → L4 E: {len(synapses_LTS4_to_L4)} связей")

synapses_LTS23_to_RS, netcons_LTS23_to_RS = connect_inh(L23_I_LTS23, L23_E_SyppyrRS)
print(f"L2/3 LTS → RS: {len(synapses_LTS23_to_RS)} связей")

synapses_Bask23_to_RS, netcons_Bask23_to_RS = connect_inh(L23_I_Bask23, L23_E_SyppyrRS)
print(f"L2/3 Basket → RS: {len(synapses_Bask23_to_RS)} связей")

synapses_nRT_to_nRT, netcons_nRT_to_nRT = connect_inh(thalamus_I_nRT, thalamus_I_nRT)
print(f"nRT → nRT: {len(synapses_nRT_to_nRT)} связей")

print("\n" + "="*60)
print("ЗАПУСК СИМУЛЯЦИИ")
print("="*60)

h.finitialize(-65)
h.continuerun(h.tstop)

print(f"✅ Симуляция завершена! Время: {h.tstop} мс")

print("\n" + "="*60)
print("ВИЗУАЛИЗАЦИЯ")
print("="*60)

# Функция для подсчёта спайков
def count_spikes(neuron, threshold=0):
    v = np.array(neuron.vvec)
    count = 0
    for i in range(1, len(v)):
        if v[i-1] < threshold and v[i] >= threshold:
            count += 1
    return count

# Сбор статистики
l4_neurons = flatten(L4_E_Spinstel4)[:50]
l4_spikes = sum(count_spikes(n) for n in l4_neurons)
l23_neurons = flatten(L23_E_SyppyrRS)[:50] + flatten(L23_E_SyppyrFRB)[:50]
l23_spikes = sum(count_spikes(n) for n in l23_neurons)

print(f"L4 нейроны (первые 50): {l4_spikes} спайков")
print(f"L2/3 нейроны (первые 100): {l23_spikes} спайков")

# График мембранного потенциала
plt.figure(figsize=(12, 6))
for i, neuron in enumerate(flatten(L4_E_Spinstel4)[:5]):
    v = neuron.vvec.as_numpy()
    t = neuron.tvec.as_numpy()
    mask = t <= 100
    plt.plot(t[mask], v[mask], alpha=0.7, label=f"L4 Neuron {i+1}")

plt.title("Мембранный потенциал нейронов L4 (~1340 нейронов)", fontsize=14)
plt.xlabel("Время (мс)")
plt.ylabel("Потенциал (мВ)")
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)
plt.savefig("simulation_result.png", dpi=150)
plt.show()

# Сохранение результатов
with open("results.txt", "w") as f:
    f.write(f"Всего нейронов: {total_neurons}\n")
    f.write(f"Время симуляции: {h.tstop} мс\n")
    f.write(f"L4 спайков (первые 50): {l4_spikes}\n")
    f.write(f"L2/3 спайков (первые 100): {l23_spikes}\n")
    f.write("Симуляция на ~1340 нейронах завершена\n")

print("\n✅ Готово! Результаты сохранены: simulation_result.png, results.txt")
