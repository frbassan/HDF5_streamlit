import h5py
import numpy as np
import time
import os

# =========================================================================
# CONFIGURAÇÕES GERAIS DA FIBRA E ARQUIVO
# =========================================================================
NOME_ARQUIVO = "Simulated_FiberTest_TSB_10km.h5"

CONFIG = {
    "comprimento_fibra_m": 10000.0,   # 10 km
    "resolucao_amostragem_m": 0.1,    # Distância entre pontos (0.1m)
    "quantidade_medicoes": 60,        # Número de traces temporais
    "intervalo_medicao_s": 60,        # 1 minuto entre cada trace
    "temp_base_c": 25.0,              # Temperatura de repouso
    "strain_base_ue": 0.0,            # Tensão de repouso (microstrain)
    "brillouin_base_mhz": 10850.0,    # Frequência central de referência
}

# =========================================================================
# CONFIGURAÇÃO DE EVENTOS (EDITE AQUI)
# =========================================================================
# Tipos de Evolução: 
#   "senoidal" (sobe e desce), 
#   "linear" (cresce constante), 
#   "pico" (acontece em um minuto específico)

EVENTOS_TEMPERATURA = [
    # { "metro": pos, "largura": m, "amplitude": ºC, "evolucao": tipo, "param": valor }
    {"metro": 2200, "largura": 3,  "amplitude": 80.0, "evolucao": "senoidal", "param": None},
    {"metro": 5300, "largura": 5,  "amplitude": 45.0, "evolucao": "senoidal", "param": None},
    {"metro": 4500, "largura": 1.5, "amplitude": 150.0, "evolucao": "pico",    "param": 40}, # Pico no minuto 40
    {"metro": 6900, "largura": 10, "amplitude": -15.0, "evolucao": "linear",  "param": None} # Resfriamento gradual
]

EVENTOS_STRAIN = [
    # { "metro": pos, "largura": m, "amplitude": µε, "evolucao": tipo, "param": valor }
    {"metro": 2400, "largura": 4,  "amplitude": 1300.0,  "evolucao": "senoidal", "param": None},
    {"metro": 5800, "largura": 8,  "amplitude": -1200.0, "evolucao": "linear",   "param": None}, # Compressão
    {"metro": 3200, "largura": 3.5, "amplitude": 800.0,   "evolucao": "onda",     "param": 2}    # 2 ciclos de vibração
]

# =========================================================================
# PROCESSAMENTO (NÃO É NECESSÁRIO EDITAR ABAIXO)
# =========================================================================

if os.path.exists(NOME_ARQUIVO): os.remove(NOME_ARQUIVO)

# Preparação de Eixos
n_distances = int(CONFIG["comprimento_fibra_m"] / CONFIG["resolucao_amostragem_m"])
distances = np.linspace(0, CONFIG["comprimento_fibra_m"], n_distances, endpoint=False)
base_timestamp = time.time()

# Sensibilidades Reais Febus
temp_sens = 1.07    # MHz / ºC
strain_sens = 0.046 # MHz / ue

# Alocação
temp_data = np.zeros((CONFIG["quantidade_medicoes"], n_distances), dtype=np.float32)
strain_data = np.zeros((CONFIG["quantidade_medicoes"], n_distances), dtype=np.float32)
bsl_data = np.zeros((CONFIG["quantidade_medicoes"], n_distances), dtype=np.float32)
start_times = np.zeros(CONFIG["quantidade_medicoes"], dtype=np.float64)

print(f"Iniciando simulação: {n_distances} pontos x {CONFIG['quantidade_medicoes']} tempos...")

for t in range(CONFIG["quantidade_medicoes"]):
    start_times[t] = base_timestamp + (t * CONFIG["intervalo_medicao_s"])
    
    # Base com ruído
    t_arr = np.full(n_distances, CONFIG["temp_base_c"]) + np.random.normal(0, 0.2, n_distances)
    s_arr = np.full(n_distances, CONFIG["strain_base_ue"]) + np.random.normal(0, 5.0, n_distances)

    # Evoluções temporais auxiliares
    progresso = t / (CONFIG["quantidade_medicoes"] - 1)
    fator_seno = np.sin(progresso * np.pi)
    
    # Processar Temperatura
    for ev in EVENTOS_TEMPERATURA:
        fator = 1.0
        if ev["evolucao"] == "senoidal": fator = fator_seno
        elif ev["evolucao"] == "linear": fator = progresso
        elif ev["evolucao"] == "pico":   fator = np.exp(-0.5 * ((t - ev["param"]) / 5)**2)
        
        # Gaussian Kernel para o Hotspot
        idx_centro = int(ev["metro"] / CONFIG["resolucao_amostragem_m"])
        largura_pts = ev["largura"] / CONFIG["resolucao_amostragem_m"]
        t_arr += ev["amplitude"] * fator * np.exp(-0.5 * ((np.arange(n_distances) - idx_centro) / largura_pts)**2)

    # Processar Strain
    for ev in EVENTOS_STRAIN:
        fator = 1.0
        if ev["evolucao"] == "senoidal": fator = fator_seno
        elif ev["evolucao"] == "linear": fator = progresso
        elif ev["evolucao"] == "onda":    fator = np.sin(progresso * 2 * np.pi * ev["param"])
        
        idx_centro = int(ev["metro"] / CONFIG["resolucao_amostragem_m"])
        largura_pts = ev["largura"] / CONFIG["resolucao_amostragem_m"]
        s_arr += ev["amplitude"] * fator * np.exp(-0.5 * ((np.arange(n_distances) - idx_centro) / largura_pts)**2)

    # Cálculo do Brillouin Shift (BSL)
    bsl_arr = CONFIG["brillouin_base_mhz"] + \
              (t_arr - CONFIG["temp_base_c"]) * temp_sens + \
              (s_arr - CONFIG["strain_base_ue"]) * strain_sens
    
    temp_data[t, :] = t_arr
    strain_data[t, :] = s_arr
    bsl_data[t, :] = bsl_arr

# Escrita HDF5
with h5py.File(NOME_ARQUIVO, 'w') as f:
    # Atributos ROOT
    attrs = {
        'fiberLength': int(CONFIG["comprimento_fibra_m"]),
        'sampling_resolution': CONFIG["resolucao_amostragem_m"],
        'start_time': start_times[0],
        'end_time': start_times[-1],
        'temp_freq_sensitivity': temp_sens,
        'strain_freq_sensitivity': strain_sens,
        'freq_fiber': CONFIG["brillouin_base_mhz"]
    }
    for k, v in attrs.items(): f.attrs[k] = v

    # Datasets
    f.create_dataset('distances', data=distances)
    f.create_dataset('start_times', data=start_times)
    f.create_dataset('temp_data', data=temp_data)
    f.create_dataset('strain_data', data=strain_data)
    f.create_dataset('bsl_data', data=bsl_data)

print(f">>> Sucesso! Arquivo '{NOME_ARQUIVO}' gerado.")