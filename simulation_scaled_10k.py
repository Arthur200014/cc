
import subprocess
import sys
import os

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

SCALE_FACTOR = 9
CONFIG_PATH = "optimization/network_realistic.yaml"

print("="*60)
print(f"КОНФИГ: {CONFIG_PATH}")
print(f"МАСШТАБИРОВАНИЕ: x{SCALE_FACTOR}")
print("="*60)

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

N_thalamus_E = config['network']['thalamus']['E'] * SCALE_FACTOR
N_thalamus_I = config['network']['thalamus']['I'] * SCALE_FACTOR
N_L4_E = config['network']['L4']['E'] * SCALE_FACTOR
N_L4_I = config['network']['L4']['I'] * SCALE_FACTOR
N_L23_E = config['network']['L23']['E'] * SCALE_FACTOR
N_L23_I = config['network']['L23']['I'] * SCALE_FACTOR
N_L5_E = config['network']['L5']['E'] * SCALE_FACTOR
N_L5_I = config['network']['L5']['I'] * SCALE_FACTOR
N_L6_E = config['network']['L6']['E'] * SCALE_FACTOR
N_L6_I = config['network']['L6']['I'] * SCALE_FACTOR

h.tstop = config['simulation']['tstop']
h.dt = config['simulation']['dt']
v_init = config['simulation']['v_init']

print(f"\nТаламус: {N_thalamus_E} E + {N_thalamus_I} I = {N_thalamus_E + N_thalamus_I}")
print(f"L4: {N_L4_E} E + {N_L4_I} I = {N_L4_E + N_L4_I}")
print(f"L2/3: {N_L23_E} E + {N_L23_I} I = {N_L23_E + N_L23_I}")
print(f"L5: {N_L5_E} E + {N_L5_I} I = {N_L5_E + N_L5_I}")
print(f"L6: {N_L6_E} E + {N_L6_I} I = {N_L6_E + N_L6_I}")

total = (N_thalamus_E + N_thalamus_I + N_L4_E + N_L4_I + 
         N_L23_E + N_L23_I + N_L5_E + N_L5_I + N_L6_E + N_L6_I)
print(f"\nВСЕГО НЕЙРОНОВ: {total}")

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

def flatten(population):
    if population and isinstance(population[0], list):
        return [neuron for subgroup in population for neuron in subgroup]
    return population

def connect_exc(source_neurons, target_neurons, e=0, tau=2.0, threshold=0, weight=0.001, delay=3.0):
    source_neurons = flatten(source_neurons) if isinstance(source_neurons[0], list) else source_neurons
    target_neurons = flatten(target_neurons) if isinstance(target_neurons[0], list) else target_neurons
    netcons, synapses = [], []
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
    source_neurons = flatten(source_neurons) if isinstance(source_neurons[0], list) else source_neurons
    target_neurons = flatten(target_neurons) if isinstance(target_neurons[0], list) else target_neurons
    netcons, synapses = [], []
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
    source_neurons = flatten(source_neurons) if isinstance(source_neurons[0], list) else source_neurons
    target_neurons = flatten(target_neurons) if isinstance(target_neurons[0], list) else target_neurons
    netcons, synapses = [], []
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

print("\nСОЗДАНИЕ НЕЙРОНОВ")
thalamus_E_TCR = [HHNeuron(inh=False) for _ in range(N_thalamus_E)]
thalamus_I_nRT = [HHNeuron(inh=True) for _ in range(N_thalamus_I)]
L4_E_Spinstel4 = [HHNeuron() for _ in range(N_L4_E)]
L4_I_LTS4 = [HHNeuron(inh=True) for _ in range(N_L4_I)]
L23_E_RS = [HHNeuron() for _ in range(N_L23_E // 2)]
L23_E_FRB = [HHNeuron() for _ in range(N_L23_E // 2)]
L23_I_Bask = [HHNeuron(inh=True) for _ in range(N_L23_I // 3)]
L23_I_LTS = [HHNeuron(inh=True) for _ in range(N_L23_I // 3)]
L23_I_Axax = [HHNeuron(inh=True) for _ in range(N_L23_I // 3)]
L5_E_TuftRS = [HHNeuron() for _ in range(N_L5_E // 2)]
L5_E_TuftIB = [HHNeuron() for _ in range(N_L5_E // 2)]
total_I_56 = N_L5_I + N_L6_I
L56_I_Bask = [HHNeuron(inh=True) for _ in range(total_I_56 // 3)]
L56_I_LTS = [HHNeuron(inh=True) for _ in range(total_I_56 // 3)]
L56_I_Axax = [HHNeuron(inh=True) for _ in range(total_I_56 // 3)]
L6_E_Nontuft = [HHNeuron() for _ in range(N_L6_E)]
print(f"Создано {total} нейронов")

print("\nСТИМУЛЯЦИЯ ТАЛАМУСА")
stimulate_count = min(100, N_thalamus_E)
for cell in thalamus_E_TCR[:stimulate_count]:
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
print(f"Стимулируется {stimulate_count} нейронов")

print("\nСОЗДАНИЕ СВЯЗЕЙ")
connect_exc_gauss(thalamus_E_TCR, L4_E_Spinstel4)
connect_exc(thalamus_E_TCR, thalamus_I_nRT)
connect_exc_gauss(L4_E_Spinstel4, L4_E_Spinstel4)
connect_exc(L4_E_Spinstel4, L4_I_LTS4)
connect_exc(L4_E_Spinstel4, L23_E_RS)
connect_exc(L4_E_Spinstel4, L23_E_FRB)
connect_exc(L23_E_RS, L23_E_RS)
connect_exc(L23_E_FRB, L23_E_FRB)
connect_exc(L23_E_RS, L5_E_TuftRS)
connect_exc(L23_E_FRB, L5_E_TuftRS)
connect_exc(L5_E_TuftRS, L5_E_TuftRS)
connect_exc(L5_E_TuftRS, L6_E_Nontuft)
connect_inh(L4_I_LTS4, L4_E_Spinstel4)
connect_inh(L23_I_LTS, L23_E_RS)
connect_inh(L23_I_Bask, L23_E_RS)
connect_inh(thalamus_I_nRT, thalamus_I_nRT)
print("Связи созданы")

print("\nЗАПУСК СИМУЛЯЦИИ")
h.finitialize(v_init)
h.continuerun(h.tstop)
print(f"Симуляция завершена! Время: {h.tstop} мс")

def count_spikes(neuron, threshold=0):
    v = np.array(neuron.vvec)
    count = 0
    for i in range(1, len(v)):
        if v[i-1] < threshold and v[i] >= threshold:
            count += 1
    return count

l4_sample = L4_E_Spinstel4[:100]
l4_spikes = sum(count_spikes(n) for n in l4_sample)
print(f"L4 спайков (первые 100): {l4_spikes}")

plt.figure(figsize=(12, 6))
for i, neuron in enumerate(L4_E_Spinstel4[:5]):
    v = neuron.vvec.as_numpy()
    t = neuron.tvec.as_numpy()
    plt.plot(t, v, alpha=0.7, label=f"L4 Neuron {i+1}")
plt.title(f"Мембранный потенциал нейронов L4\nМасштаб: x{SCALE_FACTOR} от realistic ({total} нейронов)", fontsize=14)
plt.xlabel("Время (мс)")
plt.ylabel("Потенциал (мВ)")
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)
plt.savefig(f"simulation_scaled_x{SCALE_FACTOR}.png", dpi=150)
plt.show()

with open(f"results_scaled_x{SCALE_FACTOR}.txt", "w") as f:
    f.write(f"Базовый конфиг: {CONFIG_PATH}\n")
    f.write(f"Масштаб: x{SCALE_FACTOR}\n")
    f.write(f"Всего нейронов: {total}\n")
    f.write(f"Время симуляции: {h.tstop} мс\n")
    f.write(f"L4 спайков (первые 100): {l4_spikes}\n")

print(f"\nГотово! simulation_scaled_x{SCALE_FACTOR}.png")
print(f"\n simulation_scaled_x{SCALE_FACTOR}.png")
