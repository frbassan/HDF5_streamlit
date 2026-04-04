import h5py
import numpy as np
import time
import os

# =========================================================================
# GERADOR DE DADOS DO FEBUS G2-R (MOCK DE ARQUIVOS .tsb.h5)
# =========================================================================
# Este script gera um arquivo HDF5 ultra-realista que compartilha as mesmas
# assinaturas do formato TSB gerado pelo interrogador FEBUS. Ele inclui 
# evolução temporal de temperatura e distensão da fibra elástica.
# =========================================================================

output_filename = "Simulated_FiberTest_TSB.h5"
if os.path.exists(output_filename):
    os.remove(output_filename)

# 1. PARAMETRIZAÇÃO DO ESPAÇO E TEMPO
fiber_length = 10000.0  # 10 km de fibra
sampling_res = 0.1     # Resolução óptica de amostragem em metros (0.1m)
n_distances = int(fiber_length / sampling_res)  # 100.000 pontos
distances = np.linspace(0, fiber_length, n_distances, endpoint=False)

n_times = 60           # 60 Medições de 1 hora no total
base_timestamp = time.time()
start_times = np.zeros(n_times, dtype=np.float64)
end_times = np.zeros(n_times, dtype=np.float64)

# 2. ALOCAÇÃO DAS MATRIZES PRINCIPAIS
# As matrizes 2D recebem todos os históricos: Linhas(Tempo), Colunas(Espaço)
temp_data = np.zeros((n_times, n_distances), dtype=np.float32)
strain_data = np.zeros((n_times, n_distances), dtype=np.float32)
bsl_data = np.zeros((n_times, n_distances), dtype=np.float32)

temperatures = np.zeros(n_times, dtype=np.float32)  # Temperatura ambiente do Hardware

print(f"Gerando dados da Fibra (Distâncias: {n_distances}, Tempos: {n_times})...")

# 3. GERAÇÃO DA FÍSICA (Crescimento Temporal)
temp_base = 25.0
strain_base = 0.0

# Calibração Baseada nos Atributos Reais
temp_freq_sensitivity = 1.07   # MHz / ºC
strain_freq_sensitivity = 0.046 # MHz / ue

for t_idx in range(n_times):
    # Preenchendo Metrólogia de Tempo (Aquisição fictícia de 1 minuto cada loop)
    start_times[t_idx] = base_timestamp + (t_idx * 60)
    end_times[t_idx] = start_times[t_idx] + 60.0
    temperatures[t_idx] = 24.58 + np.sin(t_idx * 0.1) * 0.5  # Temp da Placa-Mãe
    
    # 3.1 Geração de Temperatura (Ruído Fundo Gaussiano)
    ruido_termico = np.random.normal(0, 0.2, n_distances)
    temp_array = np.full(n_distances, temp_base) + ruido_termico
    
    # 3.2 Geração de Tensão/Strain (Ruído Fundo)
    ruido_mecanico = np.random.normal(0, 5.0, n_distances)
    strain_array = np.full(n_distances, strain_base) + ruido_mecanico
    
    # Fator de tempo senoidal (Sobe e Desce ao longo das 60 medições)
    # np.sin() faz um arco que sobe até o meio da hora e volta a 0 no final da hora.
    fator_tempo_sobe_desce = np.sin((t_idx / (n_times - 1)) * np.pi)
    
    # === MÚLTIPLOS HOTSPOTS ESTRATÉGICOS ===
    # Hotspot 1: Metro 1200 (Amostra 12000) - Incêndio cíclico forte
    temp_array += 80.0 * fator_tempo_sobe_desce * np.exp(-0.5 * ((np.arange(n_distances) - 12000) / 30)**2)
    
    # Hotspot 2: Metro 5500 (Amostra 55000) - Vazamento médio
    temp_array += 45.0 * fator_tempo_sobe_desce * np.exp(-0.5 * ((np.arange(n_distances) - 55000) / 50)**2)
    
    # Hotspot 3: Metro 8200 (Amostra 82000) - Pico agudo repentino (só esquenta no minuto 40)
    fator_rapido_minuto40 = np.exp(-0.5 * ((t_idx - 40) / 5)**2)
    temp_array += 150.0 * fator_rapido_minuto40 * np.exp(-0.5 * ((np.arange(n_distances) - 82000) / 15)**2)

    # === MÚLTIPLOS EVENTOS DE TENSÃO MECÂNICA ===
    # Strain 1: Metro 3400 (Tensão elástica que puxa e afrouxa com o tempo)
    strain_array += 1300.0 * fator_tempo_sobe_desce * np.exp(-0.5 * ((np.arange(n_distances) - 34000) / 40)**2)
    
    # Strain 2: Metro 6800 (Aperto constante, afunda o cabo gradualmente)
    fator_linear_subida = t_idx / (n_times - 1)
    strain_array -= 1200.0 * fator_linear_subida * np.exp(-0.5 * ((np.arange(n_distances) - 68000) / 80)**2)
    
    # Strain 3: Metro 9100 (Vibração dupla)
    fator_onda_dupla = np.sin((t_idx / (n_times - 1)) * 2 * np.pi)
    strain_array += 800.0 * fator_onda_dupla * np.exp(-0.5 * ((np.arange(n_distances) - 91000) / 35)**2)

    # 3.3 Combinando Termo-Mecânica para obter o Brillouin Shift Genuíno (BSL)
    # Frequência central base arbitrária (exemplo, não exato: 10850) + desvios
    bsl_base = 10850.0 
    bsl_array = bsl_base + (temp_array - temp_base) * temp_freq_sensitivity + (strain_array - strain_base) * strain_freq_sensitivity
    
    # Salvar nas matrizes
    temp_data[t_idx, :] = temp_array
    strain_data[t_idx, :] = strain_array
    bsl_data[t_idx, :] = bsl_array

# 4. EXPORTAÇÃO PARA O ARQUIVO .h5 ESTRUTURADO
print(f"Escrevendo arquivo: {output_filename}...")
with h5py.File(output_filename, 'w') as f:
    
    # 4.1 Criando os Atributos ROOT Oficiais 
    # (Metadados que garantem a que a máquina sinta que é o arquivo original)
    f.attrs['acq_res'] = np.array([20], dtype=np.int32)
    f.attrs['ampliPower'] = np.array([20], dtype=np.int32)
    f.attrs['average'] = np.array([1201], dtype=np.int32)
    f.attrs['channel'] = np.array([1], dtype=np.int32)
    f.attrs['end_time'] = np.array([end_times[-1]], dtype=np.float64)
    f.attrs['fiberBreak'] = np.array([int(fiber_length)], dtype=np.int32)
    f.attrs['fiberFrom'] = np.array([0], dtype=np.int32)
    f.attrs['fiberLength'] = np.array([int(fiber_length)], dtype=np.int32)
    f.attrs['fiberTo'] = np.array([int(fiber_length)], dtype=np.int32)
    f.attrs['formatVersion'] = np.array([1], dtype=np.int32)
    f.attrs['freq_fiber'] = np.array([10850], dtype=np.int32)
    f.attrs['freq_offset'] = np.array([330.954], dtype=np.float32)
    f.attrs['freq_offset_abs'] = np.array([11175.0], dtype=np.float32)
    f.attrs['freq_ref'] = np.array([10975], dtype=np.int32)
    f.attrs['freq_step'] = np.array([-1.953125], dtype=np.float32)
    f.attrs['mode'] = np.array([2], dtype=np.int32)
    f.attrs['sampling_resolution'] = np.array([sampling_res], dtype=np.float32)
    f.attrs['signal_size'] = np.array([1024], dtype=np.int32)
    f.attrs['spatial_resolution'] = np.array([5], dtype=np.float32)
    f.attrs['start_time'] = np.array([start_times[0]], dtype=np.float64)
    f.attrs['strain_amp_sensitivity'] = np.array([-0.00082], dtype=np.float32)
    f.attrs['strain_freq_sensitivity'] = np.array([strain_freq_sensitivity], dtype=np.float32)
    f.attrs['temp_amp_sensitivity'] = np.array([0.003], dtype=np.float32)
    f.attrs['temp_freq_sensitivity'] = np.array([temp_freq_sensitivity], dtype=np.float32)
    f.attrs['temperature'] = np.array([24.58], dtype=np.float32)
    f.attrs['zoneCount'] = np.array([0], dtype=np.int32)
    f.attrs['zones'] = b''
    
    # 4.2 Gravando os Datasets
    f.create_dataset('distances', data=distances, dtype='float64')
    f.create_dataset('start_times', data=start_times, dtype='float64')
    f.create_dataset('end_times', data=end_times, dtype='float64')
    f.create_dataset('temperatures', data=temperatures, dtype='float32')
    f.create_dataset('temp_data', data=temp_data, dtype='float32')
    f.create_dataset('strain_data', data=strain_data, dtype='float32')
    f.create_dataset('bsl_data', data=bsl_data, dtype='float32')
    
print(">>> Geração Concluída Com Sucesso! Arquivo ultra-realista TSB foi criado.")
