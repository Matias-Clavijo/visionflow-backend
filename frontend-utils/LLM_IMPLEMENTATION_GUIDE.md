# Event Clips API - LLM Implementation Guide

Esta guía está optimizada para que un LLM trabajando en el frontend Next.js pueda implementar correctamente la API de clips con paginación y filtros.

---

## 🎯 Objetivo

Implementar un endpoint `/api/clips` que soporte:
- ✅ Paginación basada en `page` (no `skip`)
- ✅ Filtros por fecha (`from`, `to`, `date`)
- ✅ Filtro por clase detectada (`detectedClass`)
- ✅ Filtro por dispositivo (`device`)
- ✅ Validación de parámetros con mensajes de error claros
- ✅ Respuesta estructurada con metadata de paginación

---

## 📁 Archivos a Crear/Modificar

### 1. Utility: `src/lib/clipFilters.ts` (NUEVO)

**Propósito**: Validar y parsear query parameters

**Contenido completo**:

```typescript
/**
 * src/lib/clipFilters.ts
 * Utility functions for Event Clips API filtering and validation
 */

export interface ClipFilters {
  page: number;
  limit: number;
  device?: string;
  detectedClass?: string;
  dateFilter?: {
    from?: Date;
    to?: Date;
    exact?: Date;
  };
}

export interface PaginationMetadata {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

export interface ClipFilterValidationError {
  field: string;
  message: string;
}

// Valid COCO classes (80 total)
const VALID_COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

/**
 * Parse date from ISO 8601 string or Unix timestamp
 */
function parseDate(dateStr: string): Date | null {
  // Try Unix timestamp (seconds or milliseconds)
  const timestamp = parseInt(dateStr, 10);
  if (!isNaN(timestamp) && timestamp > 0) {
    const date = timestamp > 10000000000
      ? new Date(timestamp)           // milliseconds
      : new Date(timestamp * 1000);   // seconds

    if (!isNaN(date.getTime())) return date;
  }

  // Try ISO 8601 string
  const date = new Date(dateStr);
  return !isNaN(date.getTime()) ? date : null;
}

/**
 * Parse and validate all query parameters
 * Returns: { filters, errors }
 */
export function parseAllFilters(searchParams: URLSearchParams): {
  filters: ClipFilters;
  errors: ClipFilterValidationError[];
} {
  const errors: ClipFilterValidationError[] = [];

  // --- PAGINATION ---
  let page = 1;
  const pageParam = searchParams.get('page');
  if (pageParam) {
    const parsed = parseInt(pageParam, 10);
    if (isNaN(parsed) || parsed < 1) {
      errors.push({ field: 'page', message: 'Page must be a positive integer' });
    } else {
      page = parsed;
    }
  }

  let limit = 10;
  const limitParam = searchParams.get('limit');
  if (limitParam) {
    const parsed = parseInt(limitParam, 10);
    if (isNaN(parsed) || parsed < 1) {
      errors.push({ field: 'limit', message: 'Limit must be a positive integer' });
    } else if (parsed > 100) {
      errors.push({ field: 'limit', message: 'Limit cannot exceed 100' });
    } else {
      limit = parsed;
    }
  }

  // --- DEVICE FILTER ---
  const device = searchParams.get('device') || undefined;

  // --- DETECTED CLASS FILTER ---
  let detectedClass: string | undefined;
  const classParam = searchParams.get('detectedClass');
  if (classParam) {
    const normalized = classParam.toLowerCase().trim();
    if (!VALID_COCO_CLASSES.includes(normalized)) {
      errors.push({
        field: 'detectedClass',
        message: 'Invalid class. Must be a COCO class (e.g., person, car, dog)'
      });
    } else {
      detectedClass = normalized;
    }
  }

  // --- DATE FILTERS ---
  let dateFilter: { from?: Date; to?: Date; exact?: Date } | undefined;
  const dateParam = searchParams.get('date');
  const fromParam = searchParams.get('from');
  const toParam = searchParams.get('to');

  if (dateParam) {
    // Exact date mode
    const parsed = parseDate(dateParam);
    if (!parsed) {
      errors.push({ field: 'date', message: 'Invalid date format' });
    } else {
      const startOfDay = new Date(parsed);
      startOfDay.setHours(0, 0, 0, 0);
      const endOfDay = new Date(parsed);
      endOfDay.setHours(23, 59, 59, 999);

      dateFilter = { exact: parsed, from: startOfDay, to: endOfDay };
    }
  } else if (fromParam || toParam) {
    // Range mode
    dateFilter = {};

    if (fromParam) {
      const parsed = parseDate(fromParam);
      if (!parsed) {
        errors.push({ field: 'from', message: 'Invalid from date format' });
      } else {
        dateFilter.from = parsed;
      }
    }

    if (toParam) {
      const parsed = parseDate(toParam);
      if (!parsed) {
        errors.push({ field: 'to', message: 'Invalid to date format' });
      } else {
        const endOfDay = new Date(parsed);
        endOfDay.setHours(23, 59, 59, 999);
        dateFilter.to = endOfDay;
      }
    }

    // Validate range
    if (dateFilter.from && dateFilter.to && dateFilter.from > dateFilter.to) {
      errors.push({ field: 'date_range', message: '"from" must be before "to"' });
    }
  }

  return {
    filters: { page, limit, device, detectedClass, dateFilter },
    errors
  };
}

/**
 * Build MongoDB filter object
 */
export function buildMongoFilter(filters: ClipFilters): any {
  const mongoFilter: any = { 'processor.event': true };

  if (filters.device) {
    mongoFilter.device = filters.device;
  }

  if (filters.dateFilter) {
    mongoFilter.created_at = {};
    if (filters.dateFilter.from) {
      mongoFilter.created_at.$gte = filters.dateFilter.from.toISOString();
    }
    if (filters.dateFilter.to) {
      mongoFilter.created_at.$lte = filters.dateFilter.to.toISOString();
    }
  }

  if (filters.detectedClass) {
    mongoFilter['processor.tags.class_name'] = filters.detectedClass;
  }

  return mongoFilter;
}

/**
 * Calculate pagination metadata
 */
export function calculatePagination(
  page: number,
  limit: number,
  total: number
): PaginationMetadata {
  const totalPages = Math.ceil(total / limit);
  return {
    page,
    limit,
    total,
    totalPages,
    hasNext: page < totalPages,
    hasPrev: page > 1
  };
}
```

---

### 2. Types: `src/types/clips.ts` (ACTUALIZAR)

**Agregar o actualizar estos tipos**:

```typescript
/**
 * src/types/clips.ts
 * Add these types to your existing file
 */

export interface PaginationMetadata {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

export interface ClipsAPIResponse {
  success: boolean;
  data: EventClip[];
  pagination: PaginationMetadata;
  filters?: {
    device?: string;
    detectedClass?: string;
    dateRange?: {
      from?: string;
      to?: string;
    };
  };
}

export interface ClipsAPIErrorResponse {
  success: false;
  error: string;
  message: string;
  validationErrors?: Array<{
    field: string;
    message: string;
  }>;
}
```

---

### 3. API Route: `src/app/api/clips/route.ts` (REEMPLAZAR)

**Contenido completo del endpoint**:

```typescript
/**
 * src/app/api/clips/route.ts
 * GET /api/clips - Event clips with pagination and filtering
 */

import { NextRequest, NextResponse } from 'next/server';
import clientPromise from '@/lib/mongodb';
import {
  parseAllFilters,
  buildMongoFilter,
  calculatePagination
} from '@/lib/clipFilters';
import type {
  ClipsAPIResponse,
  ClipsAPIErrorResponse
} from '@/types/clips';

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  console.log('🎬 [Clips API] Request started');

  try {
    // 1. Parse and validate query parameters
    const searchParams = request.nextUrl.searchParams;
    const { filters, errors } = parseAllFilters(searchParams);

    if (errors.length > 0) {
      console.error('❌ [Clips API] Validation errors:', errors);
      const errorResponse: ClipsAPIErrorResponse = {
        success: false,
        error: 'Validation Error',
        message: 'Invalid query parameters',
        validationErrors: errors
      };
      return NextResponse.json(errorResponse, { status: 400 });
    }

    console.log('📊 [Clips API] Filters:', {
      page: filters.page,
      limit: filters.limit,
      device: filters.device || 'all',
      detectedClass: filters.detectedClass || 'all',
      hasDateFilter: !!filters.dateFilter
    });

    // 2. Connect to MongoDB
    const client = await clientPromise;
    const db = client.db('visionflow');
    const eventsCollection = db.collection('events');

    // 3. Build MongoDB filter
    const mongoFilter = buildMongoFilter(filters);
    const skip = (filters.page - 1) * filters.limit;

    console.log('🔍 [Clips API] MongoDB filter:', JSON.stringify(mongoFilter));

    // 4. Execute queries in parallel for performance
    const [events, totalCount] = await Promise.all([
      eventsCollection
        .find(mongoFilter)
        .sort({ created_at: -1 })
        .skip(skip)
        .limit(filters.limit)
        .toArray(),
      eventsCollection.countDocuments(mongoFilter)
    ]);

    console.log(`✅ [Clips API] Found ${events.length} events (total: ${totalCount})`);

    // 5. Map to response format
    const clips = events.map((event: any) => ({
      _id: event._id.toString(),
      frame_id: event.frame_id,
      device: event.device,
      timestamp: event.timestamp,
      quality_reduction: event.quality_reduction,
      processor: event.processor,
      created_at: event.created_at,
      video_url: `/api/clips/stream/${event.frame_id}`,
    }));

    // 6. Calculate pagination metadata
    const pagination = calculatePagination(filters.page, filters.limit, totalCount);

    // 7. Build response with applied filters
    const appliedFilters: any = {};
    if (filters.device) appliedFilters.device = filters.device;
    if (filters.detectedClass) appliedFilters.detectedClass = filters.detectedClass;
    if (filters.dateFilter) {
      appliedFilters.dateRange = {
        from: filters.dateFilter.from?.toISOString(),
        to: filters.dateFilter.to?.toISOString()
      };
    }

    const response: ClipsAPIResponse = {
      success: true,
      data: clips,
      pagination,
      ...(Object.keys(appliedFilters).length > 0 && { filters: appliedFilters })
    };

    const duration = Date.now() - startTime;
    console.log(`✅ [Clips API] Completed in ${duration}ms`);

    return NextResponse.json(response);
  } catch (error) {
    console.error('❌ [Clips API] Error:', error);
    const errorResponse: ClipsAPIErrorResponse = {
      success: false,
      error: 'Internal Server Error',
      message: error instanceof Error ? error.message : 'Unknown error'
    };
    return NextResponse.json(errorResponse, { status: 500 });
  }
}
```

---

## 🧪 Testing Commands

### Test 1: Basic Pagination
```bash
curl "http://localhost:3000/api/clips?page=1&limit=10"
```

### Test 2: Filter by Device
```bash
curl "http://localhost:3000/api/clips?device=Iphone&page=1"
```

### Test 3: Filter by Detected Class
```bash
curl "http://localhost:3000/api/clips?detectedClass=person&page=1"
```

### Test 4: Filter by Date Range
```bash
curl "http://localhost:3000/api/clips?from=2025-09-01&to=2025-09-30"
```

### Test 5: Combined Filters
```bash
curl "http://localhost:3000/api/clips?page=2&limit=10&device=Iphone&detectedClass=person&from=2025-09-01&to=2025-09-30"
```

### Test 6: Validation Errors
```bash
# Invalid page
curl "http://localhost:3000/api/clips?page=-1"

# Invalid class
curl "http://localhost:3000/api/clips?detectedClass=invalid_class"

# Invalid date
curl "http://localhost:3000/api/clips?from=not-a-date"
```

---

## 📊 Response Format Examples

### ✅ Success Response

```json
{
  "success": true,
  "data": [
    {
      "_id": "68d99d5a9b3abcc72cbdbd88",
      "frame_id": "f6741e71-9756-4360-bc5f-d6ade0a5ff7f",
      "device": "Iphone",
      "timestamp": 1759092045.5222375,
      "processor": {
        "count": 1,
        "tags": [
          {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.88,
            "bbox": {
              "x": 100,
              "y": 50,
              "width": 200,
              "height": 400,
              "center_x": 200,
              "center_y": 250
            }
          }
        ],
        "event": true
      },
      "created_at": "2025-09-28T20:40:58.644+00:00",
      "video_url": "/api/clips/stream/f6741e71-9756-4360-bc5f-d6ade0a5ff7f"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 150,
    "totalPages": 15,
    "hasNext": true,
    "hasPrev": false
  },
  "filters": {
    "device": "Iphone",
    "detectedClass": "person",
    "dateRange": {
      "from": "2025-09-01T00:00:00.000Z",
      "to": "2025-09-30T23:59:59.999Z"
    }
  }
}
```

### ❌ Error Response (400 - Validation)

```json
{
  "success": false,
  "error": "Validation Error",
  "message": "Invalid query parameters",
  "validationErrors": [
    {
      "field": "page",
      "message": "Page must be a positive integer"
    },
    {
      "field": "detectedClass",
      "message": "Invalid class. Must be a COCO class (e.g., person, car, dog)"
    }
  ]
}
```

---

## 🗄️ MongoDB Indexes (IMPORTANTE)

Para performance óptima, crear estos índices:

```javascript
// Conectar a MongoDB
mongosh "mongodb+srv://your-connection-string"

// Cambiar a database
use visionflow

// Crear índices
db.events.createIndex({ "created_at": -1 });
db.events.createIndex({ "processor.event": 1 });
db.events.createIndex({ "processor.tags.class_name": 1 });
db.events.createIndex({ "device": 1 });

// Índice compuesto para queries combinadas
db.events.createIndex({
  "processor.event": 1,
  "created_at": -1,
  "device": 1,
  "processor.tags.class_name": 1
});

// Verificar índices
db.events.getIndexes();
```

---

## 🔄 Migración desde API Anterior

Si el API anterior usaba `skip` en lugar de `page`:

### Antes
```typescript
GET /api/clips?skip=10&limit=10
```

### Ahora (Equivalente)
```typescript
GET /api/clips?page=2&limit=10
```

### Conversión
```
page = (skip / limit) + 1
skip = (page - 1) * limit
```

---

## ✅ Checklist de Implementación

- [ ] Crear `src/lib/clipFilters.ts` con todo el código
- [ ] Actualizar `src/types/clips.ts` con nuevos tipos
- [ ] Reemplazar `src/app/api/clips/route.ts` con nuevo código
- [ ] Verificar que `src/lib/mongodb.ts` existe (conexión MongoDB)
- [ ] Crear índices en MongoDB
- [ ] Probar con curl todos los casos de test
- [ ] Verificar validación de errores funciona
- [ ] Verificar paginación funciona correctamente
- [ ] Verificar filtros por fecha funcionan
- [ ] Verificar filtro por clase detectada funciona
- [ ] Actualizar componentes frontend para usar nueva respuesta

---

## 🐛 Troubleshooting

### Problema: "Module not found: Can't resolve '@/lib/clipFilters'"

**Solución**: Verificar que `clipFilters.ts` está en `src/lib/clipFilters.ts`

### Problema: "Property 'processor' does not exist on type 'WithId<Document>'"

**Solución**: Usar `event.processor` con `as any` o agregar tipo correcto:
```typescript
const events = await collection.find(mongoFilter).toArray() as any[];
```

### Problema: Fechas no filtran correctamente

**Solución**: Verificar que `created_at` en MongoDB es string ISO 8601, no timestamp numérico

### Problema: Validación no rechaza clases inválidas

**Solución**: Verificar que `VALID_COCO_CLASSES` incluye todas las 80 clases COCO

---

## 📚 Query Parameters Reference

| Parámetro | Tipo | Default | Max | Descripción |
|-----------|------|---------|-----|-------------|
| `page` | number | 1 | - | Número de página (empieza en 1) |
| `limit` | number | 10 | 100 | Items por página |
| `device` | string | - | - | Nombre del dispositivo (ej: "Iphone") |
| `detectedClass` | string | - | - | Clase COCO (ej: "person", "car") |
| `from` | string | - | - | Fecha inicio (ISO 8601 o timestamp) |
| `to` | string | - | - | Fecha fin (ISO 8601 o timestamp) |
| `date` | string | - | - | Fecha exacta (sobreescribe from/to) |

---

## 🎯 Valid COCO Classes

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
couch, potted plant, bed, dining table, toilet, tv, laptop, mouse,
remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator,
book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```

---

## 💡 Pro Tips

1. **Siempre usar paginación**: Nunca intentar traer todos los documentos
2. **Límite máximo 100**: Evitar sobrecarga de servidor
3. **Crear índices**: Performance crítica para queries rápidas
4. **Validar en servidor**: Nunca confiar en input del cliente
5. **Logs detallados**: Usar console.log para debugging
6. **Parallel queries**: `Promise.all([find(), count()])` para mejor performance
7. **Reset page=1**: Cuando cambian filtros, siempre volver a página 1

---

Esta guía contiene TODO lo necesario para implementar correctamente el API de clips con paginación y filtros. No necesitas documentación adicional.
