import cv2
import mediapipe as mp
import time
import os
import sys
import math
import winsound
import numpy as np  # Suporte à criação do kernel de nitidez

# Função para garantir que o executável ache o arquivo da IA
def encontra_caminho_recurso(caminho_relativo):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, caminho_relativo)

# Configurações da API do MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

caminho_modelo = encontra_caminho_recurso('pose_landmarker_full.task')
with open(caminho_modelo, 'rb') as f:
    modelo_bytes = f.read()

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=modelo_bytes),
    running_mode=VisionRunningMode.VIDEO
)

# VARIÁVEIS DE CONTROLE DA POSTURA EM 3D (Ângulos em Graus)
ref_frontal = None  # Pitch
ref_lateral = None  # Roll
ref_rotacao = None  # Yaw
calibrado = False
tempo_inicio_postura_ruim = None  

# VARIÁVEIS PARA O RELATÓRIO
contador_alertas = 0             
tempo_total_postura_ruim = 0.0   
alerta_ativo_atualmente = False  
momento_inicio_alerta = None     

# Controle para o bip não travar a imagem da câmera
ultimo_bip = 0

# Variáveis globais para os comandos dos botões
comando_calibrar = False
comando_encerrar = False

# Configuração do CLAHE para o tratamento de contraste
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Kernel de Nitidez
kernel_nitidez = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

# FUNÇÃO QUE DETECTA CLIQUES DO MOUSE
def detectar_clique(event, x, y, flags, param):
    global comando_calibrar, comando_encerrar
    if event == cv2.EVENT_LBUTTONDOWN:  
        if largura_tela - 200 <= x <= largura_tela - 110 and 5 <= y <= 45:
            comando_calibrar = True
        if largura_tela - 100 <= x <= largura_tela - 10 and 5 <= y <= 45:
            comando_encerrar = True

webcam = cv2.VideoCapture(0)

cv2.namedWindow("Monitor de Postura")
cv2.setMouseCallback("Monitor de Postura", detectar_clique)

tempo_inicio_sessao = None

# FUNÇÃO PARA CALCULAR OS 3 ÂNGULOS DO PESCOÇO EM 3D
def calcular_angulos_3d(nariz, ombro_esq, ombro_dir):
    cx = (ombro_esq.x + ombro_dir.x) / 2
    cy = (ombro_esq.y + ombro_dir.y) / 2
    cz = (ombro_esq.z + ombro_dir.z) / 2
    
    dx = nariz.x - cx
    dy = cy - nariz.y  # Invertido porque o Y do MediaPipe cresce para baixo
    dz = cz - nariz.z  # Z negativo significa mais próximo da câmera
    
    ang_frontal = math.degrees(math.atan2(dy, abs(dz)))
    ang_lateral = math.degrees(math.atan2(dy, dx))
    ang_rotacao = math.degrees(math.atan2(abs(dx), abs(dz)))
    
    return ang_frontal, ang_lateral, ang_rotacao

with PoseLandmarker.create_from_options(options) as landmarker:
    print("APLICAÇÃO INICIADA COM MONITORAMENTO ANATÔMICO EM 3D!")
    
    while webcam.isOpened():
        sucesso, frame = webcam.read()
        if not sucesso:
            break

        frame = cv2.flip(frame, 1)
        altura_tela, largura_tela, _ = frame.shape
        
        # TRATAMENTO DE IMAGEM DA WEB-CAM
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_tratado = clahe.apply(l)
        lab_tratado = cv2.merge((l_tratado, a, b))
        frame_tratado = cv2.cvtColor(lab_tratado, cv2.COLOR_LAB2BGR)
        frame_tratado = cv2.filter2D(frame_tratado, -1, kernel_nitidez)
        frame_rgb = cv2.cvtColor(frame_tratado, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = int(webcam.get(cv2.CAP_PROP_POS_MSEC))
        
        resultado = landmarker.detect_for_video(mp_image, timestamp)

        pontos_validos = False
        n_landmark = oe_landmark = od_landmark = None
        pos_x_n = pos_y_n = pos_x_oe = pos_y_oe = pos_x_od = pos_y_od = 0

        if resultado.pose_landmarks:
            for landmarks in resultado.pose_landmarks:
                if len(landmarks) > 12:
                    n_landmark = landmarks[0]
                    oe_landmark = landmarks[11]
                    od_landmark = landmarks[12]
                    
                    pos_x_n = int(n_landmark.x * largura_tela)
                    pos_y_n = int(n_landmark.y * altura_tela)
                    pos_x_oe = int(oe_landmark.x * largura_tela)
                    pos_y_oe = int(oe_landmark.y * altura_tela)
                    pos_x_od = int(od_landmark.x * largura_tela)
                    pos_y_od = int(od_landmark.y * altura_tela)
                    
                    pontos_validos = True

                    # Desenha os feedbacks visuais na tela
                    cv2.circle(frame, (pos_x_n, pos_y_n), 6, (255, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (pos_x_oe, pos_y_oe), 6, (0, 255, 255), cv2.FILLED)
                    cv2.circle(frame, (pos_x_od, pos_y_od), 6, (0, 255, 255), cv2.FILLED)
                    
                    x_c = int((pos_x_oe + pos_x_od) / 2)
                    y_c = int((pos_y_oe + pos_y_od) / 2)
                    cv2.line(frame, (pos_x_oe, pos_y_oe), (pos_x_od, pos_y_od), (0, 255, 0), 2)
                    cv2.line(frame, (x_c, y_c), (pos_x_n, pos_y_n), (255, 0, 0), 2)

        if comando_calibrar:
            comando_calibrar = False  
            if pontos_validos:
                ref_frontal, ref_lateral, ref_rotacao = calcular_angulos_3d(n_landmark, oe_landmark, od_landmark)
                calibrado = True
                tempo_inicio_sessao = time.time()
                print(f"Calibrado! F:{ref_frontal:.1f}° | L:{ref_lateral:.1f}° | R:{ref_rotacao:.1f}°")

        postura_critica = False
        texto_status = ""
        cor_texto = (0, 0, 0) 
        
        if not calibrado:
            texto_status = "Clique em CALIBRAR para iniciar"
            cor_texto = (0, 0, 0)
        else:
            # =========================================================================
            # NOVA LÓGICA DE SEGURANÇA BLINDADA: Perda de pontos = Postura Ruim
            # =========================================================================
            postura_errada = False
            falha_IA = False

            if pontos_validos:
                # Sistema está lendo pontos, avalia os ângulos normalmente
                ang_frontal, ang_lateral, ang_rotacao = calcular_angulos_3d(n_landmark, oe_landmark, od_landmark)
                
                desvio_frontal = abs(ang_frontal - ref_frontal) > 5.0  
                desvio_lateral = abs(ang_lateral - ref_lateral) > 10.0  
                desvio_rotacao = abs(ang_rotacao - ref_rotacao) > 15.0  
                postura_errada = desvio_frontal or desvio_lateral or desvio_rotacao
            else:
                # Pontos sumiram (Oclusão/Cabeça para trás). Aciona Postura Errada imediatamente
                postura_errada = True
                falha_IA = True

            # =========================================================================

            if postura_errada:
                if tempo_inicio_postura_ruim is None:
                    tempo_inicio_postura_ruim = time.time()
                
                segundos_passados = time.time() - tempo_inicio_postura_ruim

                if segundos_passados >= 5.0:
                    postura_critica = True 
                    texto_status = "ALERTA: CORRIJA SUA POSTURA!" if not falha_IA else "ALERTA: REFERENCIA PERDIDA / POSTURA RUIM"
                    cor_texto = (0, 0, 255) 

                    if not alerta_ativo_atualmente:
                        contador_alertas += 1
                        alerta_ativo_atualmente = True
                        momento_inicio_alerta = time.time()
                        
                    if time.time() - ultimo_bip > 1.5:
                        winsound.Beep(1000, 250) 
                        ultimo_bip = time.time()
                else:
                    tempo_restante = 5 - int(segundos_passados)
                    texto_status = f"Postura Inadequada... Alerta em: {tempo_restante}s" if not falha_IA else f"Buscando referências... Alerta em: {tempo_restante}s"
                    cor_texto = (0, 165, 255) 
            else:
                tempo_inicio_postura_ruim = None
                texto_status = "Postura: OK"
                cor_texto = (0, 150, 0) 

                if alerta_ativo_atualmente:
                    tempo_duracao_alerta = time.time() - momento_inicio_alerta
                    tempo_total_postura_ruim += tempo_duracao_alerta
                    alerta_ativo_atualmente = False

        # Interface gráfica com transparência
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (largura_tela, 50), (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, texto_status, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_texto, 2)
        cv2.rectangle(frame, (largura_tela - 200, 5), (largura_tela - 110, 45), (0, 180, 0), cv2.FILLED)
        cv2.putText(frame, "CALIBRAR", (largura_tela - 185, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.rectangle(frame, (largura_tela - 100, 5), (largura_tela - 10, 45), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "SAIR", (largura_tela - 67, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

        if postura_critica:
            cv2.rectangle(frame, (0, 0), (largura_tela, altura_tela), (0, 0, 255), 15)

        cv2.imshow("Monitor de Postura", frame)

        if comando_encerrar or (cv2.waitKey(1) & 0xFF == ord('q')):
            if alerta_ativo_atualmente and momento_inicio_alerta is not None:
                tempo_total_postura_ruim += (time.time() - momento_inicio_alerta)
            break

webcam.release()
cv2.destroyAllWindows()

# PROCESSAMENTO DO RELATÓRIO FINAL
if calibrado and tempo_inicio_sessao is not None:
    tempo_total_monitorado = time.time() - tempo_inicio_sessao
    tempo_total_postura_correta = max(0.0, tempo_total_monitorado - tempo_total_postura_ruim)
    
    m_tot, s_tot = divmod(int(tempo_total_monitorado), 60)
    m_ruim, s_ruim = divmod(int(tempo_total_postura_ruim), 60)
    m_corr, s_corr = divmod(int(tempo_total_postura_correta), 60)
    
    porcentagem_ruim = (tempo_total_postura_ruim / tempo_total_monitorado) * 100 if tempo_total_monitorado > 0 else 0
    porcentagem_correta = (tempo_total_postura_correta / tempo_total_monitorado) * 100 if tempo_total_monitorado > 0 else 0

    relatorio_texto = (
        "=============================================\n"
        "       RELATÓRIO DE SAÚDE ERGONÔMICA       \n"
        "=============================================\n"
        f" Data/Hora do Fim: {time.strftime('%d/%m/%Y %H:%M:%S')}\n"
        "---------------------------------------------\n"
        f" • Tempo Total Monitorado: {m_tot}m {s_tot}s\n"
        f" • Alertas de Postura Emitidos: {contador_alertas} vez(es)\n"
        f" • Tempo em Postura Correta: {m_corr}m {s_corr}s ({porcentagem_correta:.1f}% do tempo)\n"
        f" • Tempo em Postura Inadequada: {m_ruim}m {s_ruim}s ({porcentagem_ruim:.1f}% do tempo)\n"
        "=============================================\n"
    )
    
    print("\n" + relatorio_texto)
    
    try:
        pasta_relatorios = "relatorios"
        if not os.path.exists(pasta_relatorios):
            os.makedirs(pasta_relatorios)
            
        timestamp_arquivo = time.strftime("%Y%m%d_%H%M%S")
        nome_arquivo = os.path.join(pasta_relatorios, f"relatorio_postura_{timestamp_arquivo}.txt")
        
        with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write(relatorio_texto)
        print(f"💾 Relatório histórico salvo com sucesso em '{nome_arquivo}'!")
    except Exception as e:
        print(f"❌ Erro ao salvar o arquivo de texto: {e}")
else:
    print("\nO sistema não foi calibrado. Nenhuma estatística foi gerada.")