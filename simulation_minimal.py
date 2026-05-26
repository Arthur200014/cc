# -*- coding: utf-8 -*-
"""Минимальная симуляция кортикальной сети"""

from neuron import h, gui
import random
import numpy as np
import matplotlib.pyplot as plt

h.load_file('stdrun.hoc')

# Минимальный конфиг
N_thalamus_E = 1
N_thalamus_I = 1
N_L4_E = 2
N_L4_I = 1
N_L23_E = 2
N_L23_I = 1
N_L5_E = 2
N_L5_I = 1
N_L6_E = 2
N_L6_I = 1

h.tstop = 100.0

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

# Создание нейронов
thalamus_E_TCR = [HHNeuron(inh=False) for _ in range(N_thalamus_E)]
thalamus_I_nRT = [HHNeuron(inh=True) for _ in range(N_thalamus_I)]

L23_E = [HHNeuron() for _ in range(N_L23_E)]
L23_I = [HHNeuron(inh=True) for _ in range(N_L23_I)]
L4_E = [HHNeuron() for _ in range(N_L4_E)]
L4_I = [HHNeuron(inh=True) for _ in range(N_L4_I)]
L5_E = [HHNeuron() for _ in range(N_L5_E)]
L5_I = [HHNeuron(inh=True) for _ in range(N_L5_I)]
L6_E = [HHNeuron() for _ in range(N_L6_E)]
L6_I = [HHNeuron(inh=True) for _ in range(N_L6_I)]

# Стимуляция
netstims = []
for cell in thalamus_E_TCR:
    netstim = h.NetStim()
    netstim.start = 10
    netstim.number = 5
    netstim.interval = 20
    netstim.noise = 0.5
    syn = h.ExpSyn(cell.soma(0.5))
    syn.e = 0
    syn.tau = 2.0
    nc = h.NetCon(netstim, syn)
    nc.weight[0] = 0.01
    netstims.append(netstim)

# Базовые связи
connect_exc(thalamus_E_TCR, L4_E)
connect_exc(L4_E, L23_E)
connect_exc(L23_E, L5_E)
connect_exc(L5_E, L6_E)
connect_inh(L4_I, L4_E)
connect_inh(L23_I, L23_E)
connect_inh(L5_I, L5_E)
connect_inh(L6_I, L6_E)

print(f"Всего нейронов: {len(thalamus_E_TCR) + len(thalamus_I_nRT) + len(L23_E) + len(L23_I) + len(L4_E) + len(L4_I) + len(L5_E) + len(L5_I) + len(L6_E) + len(L6_I)}")

# Запуск
h.finitialize(-65)
h.continuerun(h.tstop)

# Визуализация
plt.figure(figsize=(12, 6))
for neuron in L4_E[:2]:
    v = neuron.vvec.as_numpy()
    t = neuron.tvec.as_numpy()
    plt.plot(t, v, alpha=0.7)
plt.title("Минимальный конфиг: мембранный потенциал нейронов L4")
plt.xlabel("Время (мс)")
plt.ylabel("Потенциал (мВ)")
plt.grid(True)
plt.savefig("simulation_result.png")  # сохраняем график
plt.show()

print("✅ Симуляция завершена!")
