# VisionFlow2

[![Estado CI](https://img.shields.io/badge/estado-experimental-orange.svg)](#)
[![Licencia](https://img.shields.io/badge/licencia-Por%20definir-lightgrey.svg)](#license)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/your-org/VisionFlow2/pulls)

> **POC modular de visión por computadora en tiempo real.** Captura video, detecta objetos con YOLO11 y publica eventos/clips listos para integrarse con tableros y flujos de alertas.

## Tabla de contenidos

1. [Descripción general](#descripción-general)
2. [Visuales](#visuales)
3. [Arquitectura de alto nivel](#arquitectura-de-alto-nivel)
4. [Instalación](#instalación)
5. [Uso rápido](#uso-rápido)
6. [Manual detallado](#manual-detallado)
7. [Extensibilidad](#extensibilidad)
8. [Tecnologías utilizadas](#tecnologías-utilizadas)
9. [Contribuir](#contribuir)
10. [Ayuda y soporte](#ayuda-y-soporte)
11. [Licencia](#licencia)
12. [Agradecimientos](#agradecimientos)

## Descripción general

VisionFlow2 es una **canalización open source** que sirve como blueprint para alinear capturadores, procesadores y gestores de eventos antes de escalar a producción. El desarrollo nace bajo dos premisas:

1. **Transparencia en la aplicación de modelos**: habilitar que cualquier persona (con o sin perfil técnico) pueda intercambiar modelos de visión en cuestión de minutos, sin configuraciones extensas ni dependencias ocultas.
2. **Enfoque basado en eventos**: cada detección se trata como un “evento de interés” que puede transformarse en imágenes, clips u otros artefactos persistentes, aplicando reglas como camuflado de zonas sensibles.

Sus objetivos clave son:

- Unificar fuentes RTSP/USB/archivos sin copiar frames innecesariamente.
- Orquestar detectores Ultralytics YOLO11 según la capacidad del hardware (CPU/CUDA/MPS).
- Emitir eventos con metadatos ricos, clips MP4 y estadísticas accesibles vía HTTP/Socket.IO.
- Permitir que los usuarios modifiquen el modelo de visión según necesidades del negocio (por ejemplo, pasar de conteo de personas a detección de sustracciones) sin intervención técnica directa.

### Caso de estudio

La POC parte de un escenario simple: detectar personas dentro del área cubierta por una cámara. A partir de este caso se definieron **cuatro módulos funcionales** aplicables a proyectos más complejos:

1. **Ingesta de video**: cómo se reciben los frames desde cámaras o streams.
2. **Procesamiento**: cómo se aplica el modelo de visión y se gestionan estrategias de filtrado/salto de frames.
3. **Persistencia**: cómo se almacenan los eventos (clips, imágenes, metadatos).
4. **Interacción y visualización**: cómo el usuario observa resultados y ajusta la configuración.

Cada módulo cuenta con implementaciones por defecto (ver sección de Arquitectura) pero está pensado para ser extendido con mínima fricción.

## Visuales

- `docs/media/dashboard.png`: ejemplo de tablero web mostrando bounding boxes y métricas FPS.
- `docs/media/clip-preview.gif`: GIF del clip generado al detectar intrusiones nocturnas.

> Si aún no tienes estos archivos, puedes colocar tus propias capturas en `docs/media/` y actualizar las rutas.

## Arquitectura de alto nivel

```
           ┌──────────┐
RTSP/USB ─►│ Capturer │─┐
archivo    └──────────┘ │
                         ▼
                  SharedFramePool
                         │
                         ▼
                ┌─────────────────┐
                │   Processor(s)  │
                └─────────────────┘
                         │
        ┌────────────────┴──────────────┐
        ▼                               ▼
 Event Manager (Persistencia)     Web Server (Interacción)
```

Desglose por módulo:

- **Ingesta de video** – `RtspCapturer` gestiona conexiones, reconexiones y buffers (`src/app/core/capturers/`).
- **Procesamiento** – `ObjectDetectorYOLO11` aplica modelos Ultralytics, permite intercambiar pesos y ajustar filtros (`src/app/core/processors/`).
- **Persistencia** – `EventPoster` interpreta detecciones como eventos, genera clips, aplica mascarado y sincroniza con Backblaze B2/MongoDB (`src/app/core/events_manager/`).
- **Interacción y visualización** – `web_server.py` expone endpoints REST y eventos Socket.IO para monitoreo y tuning en tiempo real.
- **Orquestación** – `DirectOrchestrator` coordina memoria compartida y garantiza que cambiar un modelo no implique reescribir el resto del pipeline (`src/app/core/orchestrators/DirectOrchestrator.py`).

## Instalación

### Paso 0 — preparar backend (una sola vez)

```bash
git clone https://github.com/your-org/VisionFlow2.git
cd visionflow-v2
python3 -m venv venv        # opcional pero recomendado
source venv/bin/activate
pip install -r requirements.txt
```

### Paso 1 — habilitar acceso a cámara en macOS

1. Abrir **System Settings → Privacy & Security → Camera**.
2. Activar el acceso para `Terminal.app`, `iTerm2` (si lo usas) y `Python`.

### Paso 2 — probar la webcam local

```bash
cd visionflow-v2   # si no estabas en la carpeta
python3 test_webcam.py
```

La consola debe mostrar mensajes como “Webcam opened successfully” y “Frame captured successfully”.

### Paso 3 — levantar el backend

```bash
cd visionflow-v2
source venv/bin/activate
python3 src/app/main_with_web.py
```

Deja esta terminal abierta. El backend quedará disponible en `http://localhost:5001`.

### Paso 4 — preparar el frontend (una sola vez)

```bash
git clone https://github.com/your-org/trabajo-de-grado-frontend.git
cd trabajo-de-grado-frontend
npm install
```

### Paso 5 — levantar el frontend

```bash
npm run dev
```

Mantén esta terminal abierta. El frontend se sirve en `http://localhost:3000`.

### Paso 6 — validar la aplicación

1. Abrir el navegador y visitar `http://localhost:3000`.
2. Confirmar:
   - Indicador “Backend Connected” en verde.
   - Video en vivo.
   - Detecciones en tiempo real (cajas, labels, métricas).

### Paso 7 — apagar servicios

- Backend: ir a la terminal donde corre `python3 src/app/main_with_web.py` y presionar `Ctrl + C`.
- Frontend: ir a la terminal donde corre `npm run dev` y presionar `Ctrl + C`.

> Sugerencia: define variables como `OPENCV_FFMPEG_CAPTURE_OPTIONS`, `B2_APP_KEY_ID`, `B2_APP_KEY`, `MONGO_URI` antes de levantar el backend si planeas usar RTSP autenticado, almacenamiento en la nube o MongoDB.

## Uso rápido

1. **Configurar el pipeline** en `src/app/main_with_web.py`:
   ```python
   PIPELINE_CONFIG = {
       "rtsp_capturer": {
           "name": "parking_norte",
           "rtsp_url": "rtsp://usuario:pass@192.168.1.10:554/stream",
           "buffer_size": 20,
           "frame_skip": 2
       },
       "object_detector": {
           "name": "yolo11_parking",
           "model_path": "models/yolo11/yolo11s.pt",
           "confidence_threshold": 0.4,
           "filter_classes": ["person", "car"]
       },
       "video_clip_generator": {
           "name": "parking_events",
           "output_dir": "output/video_clips/parking",
           "use_cloud_storage": False,
           "use_mongodb": False
       }
   }
   ```
2. **Ejecutar**
   ```bash
   source .venv/bin/activate
   python src/app/main_with_web.py
   ```
3. **Validar**
   ```bash
   curl http://localhost:5000/health
   websocat ws://localhost:5000/socket.io/?EIO=4&transport=websocket
   ```
4. **Consumir detecciones (TypeScript)**
   ```typescript
   socket.on("detections", payload => {
     const { frame_id, processor } = payload;
     console.log(`Frame ${frame_id} → ${processor.count} objetos`);
   });
   ```

## Manual detallado

- **Configuraciones recomendadas**
  - `frame_skip` en capturadores para controlar CPU.
  - `process_every_n_frames` en detectores para balancear latencia.
  - `min_clip_cooldown` en `EventPoster` para evitar clips duplicados.
- **Comprobaciones**
  ```bash
  curl http://localhost:5000/stats
  tail -f src/app/visionflow.log
  ```
- **Escenarios de referencia**
  1. *Control de aforo*: contar personas/vehículos y publicar métricas en dashboards.
  2. *Intrusión nocturna*: bajar el umbral de confianza, generar clips y sincronizarlos con MongoDB/Backblaze.

Payload real emitido por Socket.IO:
```json
{
  "frame_id": "7c6d6a9e-5c5f-4e96",
  "processor": {
    "count": 3,
    "tags": [
      {"class_name": "person", "confidence": 0.91, "bbox": {"x": 120, "y": 88, "width": 60, "height": 180}},
      {"class_name": "car", "confidence": 0.87, "bbox": {"x": 260, "y": 100, "width": 180, "height": 140}}
    ],
    "performance": {"processing_time_ms": {"total": 38.4, "inference": 32.1}}
  },
  "metadata": {"device": "cam_norte", "timestamp": 1732559201.12}
}
```

## Extensibilidad

| Módulo       | Carpeta                        | Contrato mínimo                                    | Ejemplo de extensión                    |
|--------------|--------------------------------|----------------------------------------------------|-----------------------------------------|
| Capturer     | `src/app/core/capturers/`      | `register_output_queue`, `start`, `stop`           | `UsbCameraCapturer` con OpenCV directo  |
| Processor    | `src/app/core/processors/`     | atributo `name`, método `process(FrameData)`       | Integrar CLIP/DETR u otros modelos      |
| Event Manager| `src/app/core/events_manager/` | `register_pool(pool)`, `process(FrameDescriptor)`  | Publicar eventos en Kafka o S3          |
| Web/API      | `src/app/web_server.py`        | Blueprint Flask / namespace Socket.IO              | Endpoint `/alerts` con últimas alertas  |

Pasos sugeridos:
1. Crear el módulo siguiendo los contratos anteriores.
2. Registrarlo en `main_with_web.py` dentro de `PIPELINE_CONFIG`.
3. Documentar parámetros nuevos en README + `CLAUDE.md`.

## Tecnologías utilizadas

- **Lenguaje**: Python 3.10+
- **Frameworks**: Flask, Socket.IO
- **Visión por computadora**: OpenCV, Ultralytics YOLO11
- **Persistencia opcional**: MongoDB, Backblaze B2
- **Infraestructura**: Multiprocessing + memoria compartida (`SharedFramePool`)

## Contribuir

1. Haz un fork y crea una rama descriptiva (`feature/nuevo-capturer`).
2. Sigue las guías de estilo (PEP8) y añade docstrings/badges cuando corresponda.
3. Incluye pasos de prueba manual en tu PR (comandos, capturas o logs relevantes).
4. Abre issues usando las plantillas disponibles para bugs o nuevas ideas.

Recursos útiles:
- [Make a README](https://www.makeareadme.com/)
- [GitHub Docs: About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
- [dbader/readme-template](https://github.com/dbader/readme-template)

## Ayuda y soporte

- **Issues de GitHub**: problemas técnicos, bugs y solicitudes de features.
- **Discusiones**: comparte ideas o casos de uso (habilita la pestaña Discussions).
- **Correo**: `visionflow2-support@example.com` para consultas privadas.

## Licencia

Este proyecto se desarrolló como Trabajo de Grado en Ingeniería Informática. Consulta el archivo `LICENSE` (o contacta a los mantenedores) antes de usarlo en entornos comerciales. Al contribuir aceptas que tu aporte siga la misma licencia.

## Agradecimientos

- Equipo académico que impulsó el Trabajo de Grado.
- Comunidad de [Ultralytics](https://github.com/ultralytics/ultralytics) por liberar YOLO11.
- Autores de plantillas y guías que inspiraron este README.

---

VisionFlow2 evoluciona constantemente. Comparte mejoras, documenta tus hallazgos y ayuda a que más personas adopten arquitecturas de visión por computadora reutilizando estos componentes abiertos.
