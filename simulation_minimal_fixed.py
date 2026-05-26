# -*- coding: utf-8 -*-
"""Симуляция кортикальной сети из YAML-конфига"""

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
import yaml

h.load_file('stdrun.hoc')

# ========== ЗАГРУЗКА КОНФИГА ИЗ YAML ==========
def load_config(yaml_path="optimization/network_2k.yaml"):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
print("="*60)
print("ЗАГРУЖЕН КОНФИГ:")
print("="*60)
print(f"Таламус: E={config['network']['thalamus']['E']}, I={config['network']['thalamus']['I']}")
print(f"L4: E={config['network']['L4']['E']}, I={config['network']['L4']['I']}")
print(f"L2/3: E={config['network']['L23']['E']}, I={config['network']['L23']['I']}")
print(f"L5: E={config['network']['L5']['E']}, I={config['network']['L5']['I']}")
print(f"L6: E={config['network']['L6']['E']}, I={config['network']['L6']['I']}")
print(f"Splits: {config['splits']}")
print(f"tstop={config['simulation']['tstop']} ms, dt={config['simulation']['dt']} ms")

# Параметры из конфига
N_thalamus_E = config['network']['thalamus']['E']
N_thalamus_I = config['network']['thalamus']['I']
N_L4_E = config['network']['L4']['E']
N_L4_I = config['network']['L4']['I']
N_L23_E = config['network']['L23']['E']
N_L23_I = config['network']['L23']['I']
N_L5_E = config['network']['L5']['E']
N_L5_I = config['network']['L5']['I']
N_L6_E = config['network']['L6']['E']
N_L6_I = config['network']['L6']['I']

h.tstop = config['simulation']['tstop']
h.dt = config['simulation']['dt']

class HHNeuron:
    __slots__ = ['soma', 'inh', 'vvec', 'tvec']
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
    step = max(1, size // n_subgroups)
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

print("\n" + "="*60)
print("СОЗДАНИЕ НЕЙРОНОВ")
print("="*60)

# Таламус
thalamus_E_TCR = [HHNeuron(inh=False) for _ in range(N_thalamus_E)]
thalamus_I_nRT = [HHNeuron(inh=True) for _ in range(N_thalamus_I)]

# L4
L4_E_Spinstel4 = split_population([HHNeuron() for _ in range(N_L4_E)], config['splits']['L4_E'])
L4_I_LTS4 = split_population([HHNeuron(inh=True) for _ in range(N_L4_I)], config['splits']['L4_I'])

# L2/3
L23_I_Axax23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], config['splits']['L23_I'])
L23_I_Bask23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], config['splits']['L23_I'])
L23_I_LTS23 = split_population([HHNeuron(inh=True) for _ in range(N_L23_I)], config['splits']['L23_I'])
L23_E_SyppyrRS = split_population([HHNeuron() for _ in range(N_L23_E)], config['splits']['L23_E'])
L23_E_SyppyrFRB = split_population([HHNeuron() for _ in range(N_L23_E)], config['splits']['L23_E'])

# L5
L5_E_TuftRS5 = split_population([HHNeuron() for _ in range(N_L5_E)], config['splits']['L5_E'])
L5_E_TuftIB5 = split_population([HHNeuron() for _ in range(N_L5_E)], config['splits']['L5_E'])

# L5/6 ингибиторы
total_I_56 = N_L5_I + N_L6_I
L56_I_Bask56 = split_population([HHNeuron(inh=True) for _ in range(total_I_56)], config['splits']['L56_I'])
L56_I_Axax56 = split_population([HHNeuron(inh=True) for _ in range(total_I_56)], config['splits']['L56_I'])
L56_I_LTS56 = split_population([HHNeuron(inh=True) for _ in range(total_I_56)], config['splits']['L56_I'])

# L6
L6_E_NontuftRS6 = split_population([HHNeuron() for _ in range(N_L6_E)], config['splits']['L6_E'])

# Подсчёт общего числа
total_neurons = (len(thalamus_E_TCR) + len(thalamus_I_nRT) +
                 len(flatten(L4_E_Spinstel4)) + len(flatten(L4_I_LTS4)) +
                 len(flatten(L23_E_SyppyrRS)) + len(flatten(L23_E_SyppyrFRB)) +
                 len(flatten(L23_I_Axax23)) + len(flatten(L23_I_Bask23)) + len(flatten(L23_I_LTS23)) +
                 len(flatten(L5_E_TuftRS5)) + len(flatten(L5_E_TuftIB5)) +
                 len(flatten(L6_E_NontuftRS6)) +
                 len(flatten(L56_I_Bask56)) + len(flatten(L56_I_Axax56)) + len(flatten(L56_I_LTS56)))

print(f"✅ ВСЕГО НЕЙРОНОВ: {total_neurons}")

print("\n" + "="*60)
print("ГЕНЕРАТОРЫ СПАЙКОВ ДЛЯ ТАЛАМУСА")
print("="*60)

syn_inputs, conns, netstims = [], [], []
for cell in thalamus_E_TCR[:50]:
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

print(f"Создано {len(netstims)} NetStim")

print("\n" + "="*60)
print("СОЗДАНИЕ СВЯЗЕЙ")
print("="*60)

# Основные связи (упрощённые для скорости)
connect_exc_gauss(thalamus_E_TCR, L4_E_Spinstel4)
connect_exc(thalamus_E_TCR, thalamus_I_nRT)
connect_exc_gauss(L4_E_Spinstel4, L4_E_Spinstel4)
connect_exc(L4_E_Spinstel4, L4_I_LTS4)
connect_exc(L4_E_Spinstel4, L23_E_SyppyrRS)
connect_exc(L4_E_Spinstel4, L23_E_SyppyrFRB)
connect_exc(L23_E_SyppyrRS, L23_E_SyppyrRS)
connect_exc(L23_E_SyppyrFRB, L23_E_SyppyrFRB)
connect_exc(L23_E_SyppyrRS, L5_E_TuftRS5)
connect_exc(L23_E_SyppyrFRB, L5_E_TuftRS5)
connect_exc(L5_E_TuftRS5, L5_E_TuftRS5)
connect_exc(L5_E_TuftRS5, L6_E_NontuftRS6)
connect_inh(L4_I_LTS4, L4_E_Spinstel4)
connect_inh(L23_I_LTS23, L23_E_SyppyrRS)
connect_inh(L23_I_Bask23, L23_E_SyppyrRS)
connect_inh(thalamus_I_nRT, thalamus_I_nRT)

print("Связи созданы")

print("\n" + "="*60)
print("ЗАПУСК СИМУЛЯЦИИ")
print("="*60)

h.finitialize(config['simulation']['v_init'])
h.continuerun(h.tstop)

print(f"✅ Симуляция завершена! Время: {h.tstop} мс")

def count_spikes(neuron, threshold=0):
    v = np.array(neuron.vvec)
    count = 0
    for i in range(1, len(v)):
        if v[i-1] < threshold and v[i] >= threshold:
            count += 1
    return count

# Статистика
l4_sample = flatten(L4_E_Spinstel4)[:50]
l4_spikes = sum(count_spikes(n) for n in l4_sample)

print(f"\nL4 спайков (первые 50): {l4_spikes}")

# График
plt.figure(figsize=(12, 6))
for i, neuron in enumerate(flatten(L4_E_Spinstel4)[:5]):
    v = neuron.vvec.as_numpy()
    t = neuron.tvec.as_numpy()
    plt.plot(t, v, alpha=0.7, label=f"L4 Neuron {i+1}")

plt.title(f"Мембранный потенциал нейронов L4 ({total_neurons} нейронов)", fontsize=14)
plt.xlabel("Время (мс)")
plt.ylabel("Потенциал (мВ)")
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)
plt.savefig("simulation_result.png", dpi=150)
plt.show()

with open("results.txt", "w") as f:
    f.write(f"Конфиг: network_2k.yaml\n")
    f.write(f"Всего нейронов: {total_neurons}\n")
    f.write(f"Время симуляции: {h.tstop} мс\n")
    f.write(f"L4 спайков (первые 50): {l4_spikes}\n")

print("\n✅ Готово!")
