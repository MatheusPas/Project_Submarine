import socket
import struct
import numpy as np
import cv2
import os
from ultralytics import YOLO

# Configuração do servidor
HOST = '192.168.56.1'
PORT = 12345

# Caminho absoluto para o modelo treinado
MODEL_PATH = r"pythonCodes\\best.pt"

# Pasta para salvar os arquivos .txt com detecções
DETECTIONS_DIR = "detections"
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Verificar se o arquivo do modelo existe
if not os.path.exists(MODEL_PATH):
    print(f"Erro: O arquivo do modelo não foi encontrado em {MODEL_PATH}")
    print("Verifique se o modelo foi treinado e se o caminho está correto.")
    exit(1)

# Carregar o modelo YOLOv8 treinado
try:
    model = YOLO(MODEL_PATH)
    print(f"Modelo carregado com sucesso: {MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit(1)

# Cria o socket TCP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Aguardando conexão...")
conn, addr = server_socket.accept()
print(f"Conectado a {addr}")

# Contador de frames e variável para armazenar os últimos resultados
frame_count = 0
last_results = None

while True:
    # Recebe o tamanho da imagem
    data = conn.recv(8)
    if not data:
        print("Conexão encerrada pelo cliente.")
        break
    img_size = struct.unpack("Q", data)[0]  # Converte para inteiro (Q = uint64)

    # Recebe a imagem
    img_data = b""
    while len(img_data) < img_size:
        packet = conn.recv(img_size - len(img_data))
        if not packet:
            print("Conexão interrompida durante recebimento da imagem.")
            break
        img_data += packet

    # Converte os bytes para imagem OpenCV
    frame = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("Erro ao decodificar a imagem recebida.")
        continue

    # Incrementa o contador de frames e reinicia ao atingir 6
    frame_count += 1
    if frame_count == 11:
        frame_count = 0
        print(f"Contador de frames reiniciado no frame 6.")

    # Realizar detecção a cada 5 frames
    try:
        print(f"Processando frame {frame_count}")

        if frame_count % 10 == 0:  # Executa detecção a cada 5 frames (frames 0 e 5)
            results = model(frame)  # Realiza a detecção no frame
            last_results = results  # Armazena os resultados para uso nos próximos frames
            print(f"Detecção realizada no frame {frame_count}. Resultados: {len(results)} objetos encontrados.")
        else:
            print(f"Usando resultados do frame anterior no frame {frame_count}")

        # Desenhar as detecções e salvar anotações
        annotated_frame = frame.copy()  # Cria uma cópia do frame para desenhar
        detections_found = False

        if last_results:  # Verifica se há resultados anteriores
            for result in last_results:
                boxes = result.boxes  # Obtém as bounding boxes
                print(f"Número de bounding boxes: {len(boxes)}")

                if len(boxes) == 0:
                    print(f"Nenhuma detecção no frame {frame_count}.")
                    continue

                detections_found = True
                frame_height, frame_width = frame.shape[:2]  # Dimensões do frame

                # Criar arquivo .txt para o frame atual (se houver detecções)
                if frame_count % 5 == 0:  # Salva apenas nos frames onde a detecção é realizada
                    label_file = os.path.join(DETECTIONS_DIR, f"frame_{frame_count}.txt")
                    with open(label_file, "w") as f:
                        for box in boxes:
                            # Coordenadas da bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas absolutas
                            conf = box.conf[0]  # Confiança da detecção
                            cls = int(box.cls[0])  # Classe da detecção (0 para "Pessoa")
                            label = result.names[cls]  # Nome da classe (deve ser "Pessoa")

                            print(f"Detectado: {label} com confiança {conf:.2f} no frame {frame_count}")

                            # Calcular coordenadas normalizadas no formato YOLO
                            x_center = (x1 + x2) / 2 / frame_width
                            y_center = (y1 + y2) / 2 / frame_height
                            width = (x2 - x1) / frame_width
                            height = (y2 - y1) / frame_height

                            # Escrever no arquivo .txt
                            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                            # Desenhar o quadrado preto
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)  # Preto, espessura 2

                            # Calcular a posição do texto (abaixo da bounding box)
                            text = f"{label} ({conf:.2f})"
                            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            text_w, text_h = text_size
                            text_x = x1
                            text_y = y2 + text_h + 5  # Posiciona o texto abaixo do quadrado

                            # Desenhar o texto em amarelo
                            cv2.putText(annotated_frame, text, (text_x, text_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)  # Amarelo

                    print(f"Anotação salva em {label_file}")

                else:  # Nos frames intermediários, apenas desenha as detecções
                    for box in boxes:
                        # Coordenadas da bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Converte para inteiros
                        conf = box.conf[0]  # Confiança da detecção
                        cls = int(box.cls[0])  # Classe da detecção
                        label = result.names[cls]  # Nome da classe (deve ser "Pessoa")

                        print(f"Detectado (reutilizado): {label} com confiança {conf:.2f} no frame {frame_count}")

                        # Desenhar o quadrado preto
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)  # Preto, espessura 2

                        # Calcular a posição do texto (abaixo da bounding box)
                        text = f"{label} ({conf:.2f})"
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_w, text_h = text_size
                        text_x = x1
                        text_y = y2 + text_h + 5  # Posiciona o texto abaixo do quadrado

                        # Desenhar o texto em amarelo
                        cv2.putText(annotated_frame, text, (text_x, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)  # Amarelo

        if not detections_found:
            print(f"Nenhuma detecção válida no frame {frame_count}.")

    except Exception as e:
        print(f"Erro ao realizar detecção no frame {frame_count}: {e}")
        annotated_frame = frame  # Se houver erro, usa o frame original

    # Exibe o frame com detecções
    cv2.imshow("Recebido", annotated_frame)
    # Exibe o frame com mapa de calor (COLORMAP_JET)
    cv2.imshow("Recebido hot", cv2.applyColorMap(frame, cv2.COLORMAP_JET))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
server_socket.close()
cv2.destroyAllWindows()