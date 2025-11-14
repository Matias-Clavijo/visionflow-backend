# Live Inference Integration - LLM Implementation Guide

Esta guía explica cómo integrar tu frontend Next.js con el backend VisionFlow v2 para recibir **detecciones en tiempo real** y **métricas de performance** desde la cámara RTSP.

---

## 🎯 Objetivo

Conectar el frontend al backend Python para:
- ✅ Recibir detecciones en tiempo real vía WebSocket
- ✅ Mostrar bounding boxes sobre video/cámara
- ✅ Visualizar métricas de performance (FPS, latency, inferencia)
- ✅ Monitorear estado de conexión del backend

---

## 🏗️ Arquitectura

```
Cámara RTSP → VisionFlow Backend (Python)
                    ↓
                YOLOv4 Detection
                    ↓
              Flask/SocketIO Server (puerto 5000)
                    ↓
              WebSocket Broadcast
                    ↓
         Next.js Frontend (tu aplicación)
              ↓                ↓
     Detecciones         Performance Stats
```

---

## 📡 Backend ya está Listo

El backend VisionFlow v2 YA TIENE implementado:

### 1. Web Server en Puerto 5000
- Flask + SocketIO
- CORS habilitado
- WebSocket broadcasting automático

### 2. Endpoints REST API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Health check + estadísticas |
| `/stats` | GET | Métricas de servidor |
| `/detections/latest` | GET | Últimas detecciones (polling) |
| `/` | GET | Info del API |

### 3. Eventos WebSocket

| Evento | Dirección | Payload | Descripción |
|--------|-----------|---------|-------------|
| `connect` | Client → Server | - | Cliente conectado |
| `disconnect` | Client → Server | - | Cliente desconectado |
| `detections` | Server → Client | `Detection[]` | Detecciones en tiempo real |
| `ping` | Client → Server | - | Keep-alive |
| `pong` | Server → Client | `{ timestamp }` | Keep-alive response |

---

## 📦 Formato de Datos

### Detection Object

El backend envía detecciones en este formato (compatible con tu frontend):

```typescript
interface Detection {
  id: string;                              // "{timestamp}-{class_name}"
  class: string;                           // COCO class: "person", "car", etc.
  confidence: number;                      // 0.0 - 1.0
  bbox: [number, number, number, number]; // [x, y, width, height]
  timestamp: number;                       // Unix timestamp (milliseconds)
}
```

**Ejemplo de mensaje WebSocket**:
```json
[
  {
    "id": "1730833245123-person",
    "class": "person",
    "confidence": 0.89,
    "bbox": [100, 50, 200, 400],
    "timestamp": 1730833245123
  },
  {
    "id": "1730833245123-car",
    "class": "car",
    "confidence": 0.76,
    "bbox": [300, 200, 150, 100],
    "timestamp": 1730833245123
  }
]
```

### Health Check Response

```json
{
  "status": "healthy",
  "service": "VisionFlow v2 Backend",
  "version": "2.0.0",
  "uptime_seconds": 10.5,
  "stats": {
    "total_frames_processed": 42,
    "total_detections": 15,
    "connected_clients": 1
  }
}
```

---

## 🚀 Implementación Frontend

### Step 1: Variables de Entorno

Crear o actualizar `.env.local`:

```bash
# Backend WebSocket URL
NEXT_PUBLIC_WS_URL=http://localhost:5000

# Optional: Backend REST API URL
NEXT_PUBLIC_API_URL=http://localhost:5000
```

---

### Step 2: Hook useWebSocket (YA LO TIENES)

Tu hook `useWebSocket` ya está perfecto y compatible. Solo necesitas conectarlo:

```typescript
// src/hooks/useWebSocket.ts (YA LO TIENES)
// No necesitas modificar nada, ya funciona con el backend
```

**Verifica que tengas**:
- `socket.on('detections', callback)` ✅ (ya lo tienes)
- `connect()` method ✅ (ya lo tienes)
- Auto-reconnect logic ✅ (ya lo tienes)

---

### Step 3: Conectar al Backend

#### 3.1. En tu componente principal

```typescript
// src/app/page.tsx o donde tengas tu componente principal

'use client';

import { useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useDetections } from '@/hooks/useDetections';

export default function Home() {
  const webSocket = useWebSocket();
  const detections = useDetections();

  // Conectar al backend al montar el componente
  useEffect(() => {
    webSocket.connect(); // Conecta a NEXT_PUBLIC_WS_URL

    // Configurar listener para detecciones
    webSocket.onDetections((newDetections) => {
      console.log('🎯 Detecciones recibidas:', newDetections);

      // Actualizar estado local con las detecciones
      // (tu hook useDetections ya maneja esto)
    });

    return () => {
      webSocket.disconnect();
      webSocket.offDetections();
    };
  }, []);

  return (
    <div>
      {/* Estado de conexión */}
      <ConnectionStatus
        isConnected={webSocket.isConnected}
        isConnecting={webSocket.isConnecting}
        error={webSocket.error}
        connectionAttempts={webSocket.connectionAttempts}
        maxAttempts={webSocket.maxAttempts}
      />

      {/* Video stream con detecciones */}
      <VideoStream detections={detections.detections} />

      {/* Panel de estadísticas */}
      <StatsPanel stats={detections.stats} />
    </div>
  );
}
```

---

### Step 4: Componente de Estado de Conexión

```typescript
// src/components/ConnectionStatus.tsx

interface ConnectionStatusProps {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  connectionAttempts: number;
  maxAttempts: number;
}

export function ConnectionStatus({
  isConnected,
  isConnecting,
  error,
  connectionAttempts,
  maxAttempts
}: ConnectionStatusProps) {
  return (
    <div className="connection-status">
      {/* Indicador visual */}
      <div className={`status-indicator ${
        isConnected
          ? 'bg-green-500 animate-pulse'
          : isConnecting
            ? 'bg-yellow-500 animate-spin'
            : 'bg-red-500'
      }`} />

      {/* Texto de estado */}
      <div className="status-text">
        {isConnected && (
          <span className="text-green-600">
            Backend Conectado - Detecciones en tiempo real activas
          </span>
        )}

        {isConnecting && (
          <span className="text-yellow-600">
            Conectando al backend... ({connectionAttempts}/{maxAttempts})
          </span>
        )}

        {!isConnected && !isConnecting && (
          <span className="text-red-600">
            Backend Desconectado - {error || 'Usando modo simulación'}
          </span>
        )}
      </div>

      {/* Botón de reconexión manual */}
      {!isConnected && !isConnecting && (
        <button
          onClick={() => {
            // Resetear y reconectar
            webSocket.disconnect();
            setTimeout(() => webSocket.connect(), 200);
          }}
          className="px-3 py-1.5 bg-blue-500 text-white rounded"
        >
          Reconectar Backend
        </button>
      )}
    </div>
  );
}
```

---

### Step 5: Hook para Performance Stats

Crear un hook para consultar métricas del backend:

```typescript
// src/hooks/useBackendStats.ts

import { useState, useEffect } from 'react';

interface BackendStats {
  uptime_seconds: number;
  total_frames_processed: number;
  total_detections: number;
  connected_clients: number;
  avg_detections_per_frame: number;
}

interface UseBackendStatsReturn {
  stats: BackendStats | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useBackendStats(
  refreshInterval: number = 5000 // Actualizar cada 5 segundos
): UseBackendStatsReturn {
  const [stats, setStats] = useState<BackendStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_WS_URL}/stats`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stats');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();

    // Auto-refresh
    const interval = setInterval(fetchStats, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  return {
    stats,
    loading,
    error,
    refresh: fetchStats
  };
}
```

---

### Step 6: Panel de Performance Stats

```typescript
// src/components/PerformancePanel.tsx

import { useBackendStats } from '@/hooks/useBackendStats';

export function PerformancePanel() {
  const { stats, loading, error } = useBackendStats(5000); // Refresh cada 5s

  if (loading && !stats) {
    return <div>Cargando estadísticas...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!stats) {
    return null;
  }

  return (
    <div className="performance-panel">
      <h3>Performance del Backend</h3>

      <div className="stats-grid">
        {/* Uptime */}
        <div className="stat-card">
          <div className="stat-label">Uptime</div>
          <div className="stat-value">
            {formatUptime(stats.uptime_seconds)}
          </div>
        </div>

        {/* Frames procesados */}
        <div className="stat-card">
          <div className="stat-label">Frames Procesados</div>
          <div className="stat-value">
            {stats.total_frames_processed.toLocaleString()}
          </div>
        </div>

        {/* Detecciones totales */}
        <div className="stat-card">
          <div className="stat-label">Detecciones Totales</div>
          <div className="stat-value">
            {stats.total_detections.toLocaleString()}
          </div>
        </div>

        {/* Promedio detecciones por frame */}
        <div className="stat-card">
          <div className="stat-label">Avg. Detecciones/Frame</div>
          <div className="stat-value">
            {stats.avg_detections_per_frame.toFixed(2)}
          </div>
        </div>

        {/* Clientes conectados */}
        <div className="stat-card">
          <div className="stat-label">Clientes Conectados</div>
          <div className="stat-value">
            {stats.connected_clients}
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper para formatear uptime
function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  return `${hours}h ${minutes}m ${secs}s`;
}
```

---

### Step 7: Integrar Detecciones con tu VideoStream

Tu componente `VideoStream` ya está preparado, solo necesitas asegurarte de que las detecciones se pasen correctamente:

```typescript
// src/components/VideoStream.tsx (YA LO TIENES, solo verificar)

interface VideoStreamProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  detections: Detection[]; // ← Aquí llegan las detecciones del WebSocket
  isStreaming: boolean;
  error: string | null;
  onStartStream: () => void;
  onStopStream: () => void;
}

export function VideoStream({
  videoRef,
  detections, // ← Recibe detecciones del backend
  isStreaming,
  error,
  onStartStream,
  onStopStream
}: VideoStreamProps) {
  return (
    <div className="video-container">
      <video ref={videoRef} autoPlay playsInline />

      {/* Overlay de detecciones */}
      <DetectionOverlay
        detections={detections}
        videoWidth={videoRef.current?.videoWidth || 0}
        videoHeight={videoRef.current?.videoHeight || 0}
      />
    </div>
  );
}
```

---

## 🎨 Ejemplo Completo: Página con Live Inference

```typescript
// src/app/live-inference/page.tsx

'use client';

import { useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useDetections } from '@/hooks/useDetections';
import { useWebRTC } from '@/hooks/useWebRTC';
import { VideoStream } from '@/components/VideoStream';
import { ConnectionStatus } from '@/components/ConnectionStatus';
import { PerformancePanel } from '@/components/PerformancePanel';
import { DetectionsList } from '@/components/DetectionsList';

export default function LiveInferencePage() {
  const webSocket = useWebSocket();
  const detections = useDetections();
  const webRTC = useWebRTC();

  // Conectar al backend al montar
  useEffect(() => {
    console.log('🚀 Conectando al backend VisionFlow...');
    webSocket.connect();

    // Escuchar detecciones en tiempo real
    webSocket.onDetections((newDetections) => {
      console.log('🎯 Detecciones recibidas:', newDetections.length);

      // Las detecciones ya vienen en el formato correcto
      // Tu hook useDetections puede manejarlas directamente
      // O puedes procesarlas aquí
    });

    return () => {
      console.log('🛑 Desconectando del backend');
      webSocket.disconnect();
      webSocket.offDetections();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 p-4">
        <h1 className="text-2xl font-bold">
          VisionFlow Live Inference
        </h1>
        <p className="text-gray-400">
          Detección de objetos en tiempo real con YOLOv4
        </p>
      </header>

      {/* Connection Status */}
      <div className="p-4 bg-gray-800 border-b border-gray-700">
        <ConnectionStatus
          isConnected={webSocket.isConnected}
          isConnecting={webSocket.isConnecting}
          error={webSocket.error}
          connectionAttempts={webSocket.connectionAttempts}
          maxAttempts={webSocket.maxAttempts}
        />
      </div>

      <div className="container mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Video Stream (col-span-2) */}
          <div className="lg:col-span-2">
            <VideoStream
              videoRef={webRTC.videoRef}
              detections={detections.detections}
              isStreaming={webRTC.isStreaming}
              error={webRTC.error}
              onStartStream={webRTC.startStream}
              onStopStream={webRTC.stopStream}
            />
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* Performance Stats */}
            <PerformancePanel />

            {/* Lista de detecciones */}
            <DetectionsList detections={detections.detections} />
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## 🧪 Testing

### 1. Verificar Backend está Corriendo

```bash
# Terminal 1: Iniciar backend
cd /path/to/visionflow-v2
./start.sh

# Debe mostrar:
# ✓ Web server started successfully
#    - REST API: http://0.0.0.0:5000
#    - WebSocket: ws://0.0.0.0:5000
```

### 2. Test Health Endpoint

```bash
curl http://localhost:5000/health

# Respuesta esperada:
# {
#   "status": "healthy",
#   "service": "VisionFlow v2 Backend",
#   "version": "2.0.0",
#   "uptime_seconds": 10.5,
#   "stats": {
#     "total_frames_processed": 42,
#     "total_detections": 15,
#     "connected_clients": 0
#   }
# }
```

### 3. Test WebSocket Connection

```bash
# Instalar websocat (si no lo tienes)
brew install websocat

# Conectar al WebSocket
websocat ws://localhost:5000/socket.io/?EIO=4&transport=websocket

# Deberías ver mensajes de conexión y detecciones
```

### 4. Test Frontend

```bash
# Terminal 2: Iniciar frontend
cd /path/to/your-nextjs-app
npm run dev

# Abrir: http://localhost:3000
# Deberías ver:
# - "Backend Conectado" (indicador verde)
# - Detecciones apareciendo en tiempo real
# - Bounding boxes sobre el video
```

---

## 📊 Datos de Performance Disponibles

### Desde WebSocket (tiempo real)

Cada detección incluye:
```typescript
{
  id: string;          // Unique ID
  class: string;       // Clase detectada
  confidence: number;  // Confianza del modelo
  bbox: [x, y, w, h]; // Posición del objeto
  timestamp: number;   // Timestamp del frame
}
```

### Desde REST API `/stats` (polling)

```typescript
{
  uptime_seconds: number;              // Tiempo que lleva corriendo
  total_frames_processed: number;      // Total de frames procesados
  total_detections: number;            // Total de objetos detectados
  connected_clients: number;           // Clientes WebSocket conectados
  avg_detections_per_frame: number;    // Promedio de detecciones por frame
}
```

### Desde MongoDB (histórico)

Cada evento guardado incluye métricas de performance:

```typescript
{
  processor: {
    performance: {
      processing_time_ms: {
        total: 45.23,        // Tiempo total de procesamiento
        preprocess: 2.1,     // Tiempo de preprocesamiento
        inference: 35.8,     // Tiempo de inferencia YOLOv4
        postprocess: 7.33    // Tiempo de postprocesamiento
      }
    }
  }
}
```

---

## 🎯 Métricas Recomendadas para Mostrar

### Dashboard Principal

1. **Estado de Conexión**
   - Indicador visual (verde/amarillo/rojo)
   - Texto descriptivo
   - Botón de reconexión

2. **Performance en Tiempo Real**
   - FPS actual (calcular desde timestamps)
   - Latencia promedio
   - Detecciones por segundo

3. **Estadísticas del Backend**
   - Uptime
   - Frames procesados
   - Detecciones totales
   - Clientes conectados

4. **Detecciones Actuales**
   - Lista de objetos detectados
   - Confianza de cada detección
   - Timestamp

### Ejemplo de Cálculo de FPS

```typescript
// src/hooks/useFPS.ts

import { useState, useEffect, useRef } from 'react';
import type { Detection } from '@/types/detection';

export function useFPS(detections: Detection[]) {
  const [fps, setFps] = useState(0);
  const frameTimestamps = useRef<number[]>([]);

  useEffect(() => {
    if (detections.length === 0) return;

    const now = Date.now();
    frameTimestamps.current.push(now);

    // Mantener solo últimos 60 timestamps (1 segundo a 60 FPS)
    if (frameTimestamps.current.length > 60) {
      frameTimestamps.current.shift();
    }

    // Calcular FPS
    if (frameTimestamps.current.length >= 2) {
      const first = frameTimestamps.current[0];
      const last = frameTimestamps.current[frameTimestamps.current.length - 1];
      const duration = (last - first) / 1000; // segundos

      if (duration > 0) {
        const calculatedFps = frameTimestamps.current.length / duration;
        setFps(Math.round(calculatedFps));
      }
    }
  }, [detections]);

  return fps;
}
```

---

## 🐛 Troubleshooting

### Problema: Frontend no conecta al backend

**Solución**:
1. Verificar que el backend está corriendo: `curl http://localhost:5000/health`
2. Verificar `NEXT_PUBLIC_WS_URL` en `.env.local`
3. Verificar CORS en el backend (ya está habilitado por defecto)
4. Revisar console del navegador para errores de WebSocket

### Problema: Detecciones no aparecen

**Solución**:
1. Verificar que la cámara RTSP está conectada
2. Revisar logs del backend: `tail -f logs/visionflow.log`
3. Verificar que el evento `detections` está registrado correctamente
4. Confirmar que hay objetos en el campo de visión de la cámara

### Problema: Performance baja / lag

**Solución**:
1. Ajustar `broadcast_every_n_frames` en el backend (default: 2)
2. Reducir resolución de captura RTSP (640x480 recomendado)
3. Aumentar `frame_skip` en el capturer
4. Verificar uso de CPU/GPU en el servidor

### Problema: WebSocket se desconecta frecuentemente

**Solución**:
1. Implementar ping/pong keep-alive (ya está en el hook)
2. Aumentar timeout del WebSocket
3. Verificar estabilidad de la red
4. Revisar logs del backend para errores

---

## ✅ Checklist de Implementación

- [ ] Backend VisionFlow está corriendo en puerto 5000
- [ ] `.env.local` tiene `NEXT_PUBLIC_WS_URL=http://localhost:5000`
- [ ] Hook `useWebSocket` está implementado (ya lo tienes)
- [ ] Componente llama `webSocket.connect()` al montar
- [ ] Listener `webSocket.onDetections()` está configurado
- [ ] Componente `ConnectionStatus` muestra estado
- [ ] Componente `PerformancePanel` muestra stats
- [ ] Bounding boxes se dibujan sobre el video
- [ ] Health endpoint `/health` responde correctamente
- [ ] WebSocket recibe mensajes `detections`
- [ ] Cleanup en `useEffect` desconecta al desmontar

---

## 🚀 Próximos Pasos (Opcional)

1. **Grabar Detecciones**
   - Guardar detecciones en IndexedDB del navegador
   - Exportar a CSV/JSON

2. **Filtros en Tiempo Real**
   - Mostrar solo ciertas clases (ej: solo "person")
   - Threshold de confianza ajustable

3. **Alertas**
   - Notificaciones del navegador cuando se detecta algo
   - Sonido de alerta

4. **Heatmap**
   - Visualizar zonas más frecuentemente detectadas
   - Usar canvas para overlay de heatmap

5. **Dashboard Avanzado**
   - Gráficos de detecciones por tiempo (Chart.js)
   - Histograma de clases detectadas
   - Timeline de eventos

---

## 📚 Recursos Adicionales

- **Backend Docs**: Ver `CLAUDE.md` en visionflow-v2
- **WebSocket Events**: Ver `web_server.py` para eventos disponibles
- **Detection Format**: Ver `convert_to_frontend_format()` en `web_server.py`
- **Performance Metrics**: Ver `EventPoster` para métricas detalladas

---

Esta guía contiene TODO lo necesario para implementar inferencia en vivo con detecciones y métricas de performance. El backend YA ESTÁ LISTO, solo necesitas conectar el frontend siguiendo estos pasos.
