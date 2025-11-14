# Event Clips API - Usage Examples

## Base URL
```
http://localhost:3000/api/clips
```

## Query Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `page` | number | Page number (starts at 1) | `page=2` |
| `limit` | number | Items per page (max 100) | `limit=20` |
| `device` | string | Filter by device name | `device=Iphone` |
| `detectedClass` | string | Filter by detected class (COCO) | `detectedClass=person` |
| `from` | string | Start date (ISO 8601 or timestamp) | `from=2025-09-01` |
| `to` | string | End date (ISO 8601 or timestamp) | `to=2025-09-30` |
| `date` | string | Exact date (overrides from/to) | `date=2025-09-15` |

## Response Format

### Success Response
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
              "height": 400
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

### Error Response (400 - Validation Error)
```json
{
  "success": false,
  "error": "Validation Error",
  "message": "Invalid query parameters",
  "validationErrors": [
    {
      "field": "page",
      "message": "Page must be a positive integer greater than 0"
    },
    {
      "field": "detectedClass",
      "message": "Invalid class name. Must be one of the 80 COCO classes"
    }
  ]
}
```

### Error Response (500 - Server Error)
```json
{
  "success": false,
  "error": "Internal Server Error",
  "message": "Error details here"
}
```

## Usage Examples

### 1. Basic Pagination

#### Request
```bash
GET /api/clips?page=1&limit=10
```

#### cURL
```bash
curl "http://localhost:3000/api/clips?page=1&limit=10"
```

#### JavaScript/TypeScript (Frontend)
```typescript
const response = await fetch('/api/clips?page=1&limit=10');
const data = await response.json();

console.log(`Total clips: ${data.pagination.total}`);
console.log(`Total pages: ${data.pagination.totalPages}`);
console.log(`Has next page: ${data.pagination.hasNext}`);
```

---

### 2. Filter by Device

#### Request
```bash
GET /api/clips?device=Iphone&page=1&limit=10
```

#### cURL
```bash
curl "http://localhost:3000/api/clips?device=Iphone&page=1&limit=10"
```

#### JavaScript/TypeScript
```typescript
const fetchClipsByDevice = async (device: string, page = 1) => {
  const response = await fetch(`/api/clips?device=${device}&page=${page}&limit=10`);
  return response.json();
};

const clips = await fetchClipsByDevice('Iphone', 1);
```

---

### 3. Filter by Detected Class

#### Request
```bash
GET /api/clips?detectedClass=person&page=1&limit=10
```

#### cURL
```bash
curl "http://localhost:3000/api/clips?detectedClass=person&page=1&limit=10"
```

#### JavaScript/TypeScript
```typescript
const fetchClipsByClass = async (className: string) => {
  const response = await fetch(`/api/clips?detectedClass=${className}&page=1&limit=10`);
  return response.json();
};

// Find all clips with detected persons
const personClips = await fetchClipsByClass('person');

// Find all clips with detected cars
const carClips = await fetchClipsByClass('car');
```

---

### 4. Filter by Date Range

#### Request (ISO 8601)
```bash
GET /api/clips?from=2025-09-01&to=2025-09-30&page=1
```

#### Request (Unix Timestamps)
```bash
GET /api/clips?from=1725148800&to=1727740799&page=1
```

#### cURL
```bash
curl "http://localhost:3000/api/clips?from=2025-09-01&to=2025-09-30&page=1"
```

#### JavaScript/TypeScript
```typescript
const fetchClipsByDateRange = async (from: string, to: string) => {
  const params = new URLSearchParams({
    from,
    to,
    page: '1',
    limit: '10'
  });

  const response = await fetch(`/api/clips?${params}`);
  return response.json();
};

// Using ISO dates
const clips = await fetchClipsByDateRange('2025-09-01', '2025-09-30');

// Using Date objects
const fromDate = new Date('2025-09-01').toISOString().split('T')[0];
const toDate = new Date('2025-09-30').toISOString().split('T')[0];
const clips2 = await fetchClipsByDateRange(fromDate, toDate);
```

---

### 5. Filter by Exact Date

#### Request
```bash
GET /api/clips?date=2025-09-15&page=1
```

#### cURL
```bash
curl "http://localhost:3000/api/clips?date=2025-09-15&page=1"
```

#### JavaScript/TypeScript
```typescript
const fetchClipsByDate = async (date: string) => {
  const response = await fetch(`/api/clips?date=${date}&page=1&limit=10`);
  return response.json();
};

// Get all clips from September 15, 2025
const clips = await fetchClipsByDate('2025-09-15');
```

---

### 6. Combined Filters

#### Request
```bash
GET /api/clips?page=2&limit=20&device=Iphone&detectedClass=person&from=2025-09-01&to=2025-09-30
```

#### cURL
```bash
curl "http://localhost:3000/api/clips?page=2&limit=20&device=Iphone&detectedClass=person&from=2025-09-01&to=2025-09-30"
```

#### JavaScript/TypeScript
```typescript
interface ClipFilters {
  page?: number;
  limit?: number;
  device?: string;
  detectedClass?: string;
  from?: string;
  to?: string;
  date?: string;
}

const fetchClipsWithFilters = async (filters: ClipFilters) => {
  const params = new URLSearchParams();

  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined) {
      params.append(key, String(value));
    }
  });

  const response = await fetch(`/api/clips?${params}`);
  return response.json();
};

// Usage
const clips = await fetchClipsWithFilters({
  page: 2,
  limit: 20,
  device: 'Iphone',
  detectedClass: 'person',
  from: '2025-09-01',
  to: '2025-09-30'
});
```

---

### 7. React Component with Pagination

```typescript
import { useState, useEffect } from 'react';

interface ClipsResponse {
  success: boolean;
  data: any[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export function EventClipsTable() {
  const [clips, setClips] = useState<any[]>([]);
  const [pagination, setPagination] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [filters, setFilters] = useState({
    page: 1,
    limit: 10,
    device: '',
    detectedClass: '',
    from: '',
    to: ''
  });

  const fetchClips = async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value) params.append(key, String(value));
      });

      const response = await fetch(`/api/clips?${params}`);
      const data: ClipsResponse = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch clips');
      }

      setClips(data.data);
      setPagination(data.pagination);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchClips();
  }, [filters]);

  const handleNextPage = () => {
    if (pagination?.hasNext) {
      setFilters(prev => ({ ...prev, page: prev.page + 1 }));
    }
  };

  const handlePrevPage = () => {
    if (pagination?.hasPrev) {
      setFilters(prev => ({ ...prev, page: prev.page - 1 }));
    }
  };

  return (
    <div>
      {/* Filters */}
      <div className="filters">
        <input
          type="text"
          placeholder="Device"
          value={filters.device}
          onChange={(e) => setFilters(prev => ({ ...prev, device: e.target.value, page: 1 }))}
        />
        <input
          type="text"
          placeholder="Detected Class"
          value={filters.detectedClass}
          onChange={(e) => setFilters(prev => ({ ...prev, detectedClass: e.target.value, page: 1 }))}
        />
        <input
          type="date"
          value={filters.from}
          onChange={(e) => setFilters(prev => ({ ...prev, from: e.target.value, page: 1 }))}
        />
        <input
          type="date"
          value={filters.to}
          onChange={(e) => setFilters(prev => ({ ...prev, to: e.target.value, page: 1 }))}
        />
      </div>

      {/* Results */}
      {loading && <p>Loading...</p>}
      {error && <p>Error: {error}</p>}

      {clips.length > 0 && (
        <div>
          <ul>
            {clips.map(clip => (
              <li key={clip._id}>
                {clip.device} - {clip.processor.tags[0]?.class_name} - {clip.created_at}
              </li>
            ))}
          </ul>

          {/* Pagination */}
          <div className="pagination">
            <button onClick={handlePrevPage} disabled={!pagination?.hasPrev}>
              Previous
            </button>
            <span>
              Page {pagination?.page} of {pagination?.totalPages}
            </span>
            <button onClick={handleNextPage} disabled={!pagination?.hasNext}>
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## Valid COCO Classes (for `detectedClass` filter)

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

## Performance Tips

1. **Use Pagination**: Always paginate results for better performance
2. **Limit Results**: Keep `limit` reasonable (10-50 items)
3. **Index MongoDB**: Ensure indexes on `created_at` and `processor.tags.class_name`
4. **Date Ranges**: Use specific date ranges instead of querying all data
5. **Cache Results**: Consider caching frequently accessed pages

## MongoDB Indexes (Recommended)

```javascript
// In MongoDB shell
db.events.createIndex({ "created_at": -1 });
db.events.createIndex({ "processor.event": 1 });
db.events.createIndex({ "processor.tags.class_name": 1 });
db.events.createIndex({ "device": 1 });

// Compound index for common queries
db.events.createIndex({
  "processor.event": 1,
  "created_at": -1,
  "device": 1
});
```
