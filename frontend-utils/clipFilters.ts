/**
 * Utility functions for Event Clips API - Query Parameter Validation and Filtering
 * Used by Next.js API route: /api/clips/route.ts
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

/**
 * Parse and validate pagination parameters
 */
export function parsePaginationParams(searchParams: URLSearchParams): {
  page: number;
  limit: number;
  errors: ClipFilterValidationError[];
} {
  const errors: ClipFilterValidationError[] = [];

  // Parse page (default: 1)
  let page = 1;
  const pageParam = searchParams.get('page');
  if (pageParam) {
    const parsed = parseInt(pageParam, 10);
    if (isNaN(parsed) || parsed < 1) {
      errors.push({
        field: 'page',
        message: 'Page must be a positive integer greater than 0'
      });
    } else {
      page = parsed;
    }
  }

  // Parse limit (default: 10, max: 100)
  let limit = 10;
  const limitParam = searchParams.get('limit');
  if (limitParam) {
    const parsed = parseInt(limitParam, 10);
    if (isNaN(parsed) || parsed < 1) {
      errors.push({
        field: 'limit',
        message: 'Limit must be a positive integer greater than 0'
      });
    } else if (parsed > 100) {
      errors.push({
        field: 'limit',
        message: 'Limit cannot exceed 100 items per page'
      });
    } else {
      limit = parsed;
    }
  }

  return { page, limit, errors };
}

/**
 * Parse and validate date filters
 * Accepts: ISO 8601 strings or Unix timestamps (seconds)
 */
export function parseDateFilters(searchParams: URLSearchParams): {
  dateFilter?: {
    from?: Date;
    to?: Date;
    exact?: Date;
  };
  errors: ClipFilterValidationError[];
} {
  const errors: ClipFilterValidationError[] = [];
  let dateFilter: { from?: Date; to?: Date; exact?: Date } | undefined;

  // Check for exact date first
  const dateParam = searchParams.get('date');
  if (dateParam) {
    const parsed = parseDate(dateParam);
    if (!parsed) {
      errors.push({
        field: 'date',
        message: 'Invalid date format. Use ISO 8601 (YYYY-MM-DD) or Unix timestamp'
      });
    } else {
      // For exact date, set from to start of day and to to end of day
      const startOfDay = new Date(parsed);
      startOfDay.setHours(0, 0, 0, 0);

      const endOfDay = new Date(parsed);
      endOfDay.setHours(23, 59, 59, 999);

      dateFilter = {
        exact: parsed,
        from: startOfDay,
        to: endOfDay
      };
    }

    // If exact date is provided, ignore from/to params
    return { dateFilter, errors };
  }

  // Parse from and to dates
  const fromParam = searchParams.get('from');
  const toParam = searchParams.get('to');

  if (fromParam || toParam) {
    dateFilter = {};

    if (fromParam) {
      const parsed = parseDate(fromParam);
      if (!parsed) {
        errors.push({
          field: 'from',
          message: 'Invalid from date format. Use ISO 8601 (YYYY-MM-DD) or Unix timestamp'
        });
      } else {
        dateFilter.from = parsed;
      }
    }

    if (toParam) {
      const parsed = parseDate(toParam);
      if (!parsed) {
        errors.push({
          field: 'to',
          message: 'Invalid to date format. Use ISO 8601 (YYYY-MM-DD) or Unix timestamp'
        });
      } else {
        // Set to end of day for 'to' date
        const endOfDay = new Date(parsed);
        endOfDay.setHours(23, 59, 59, 999);
        dateFilter.to = endOfDay;
      }
    }

    // Validate from < to
    if (dateFilter.from && dateFilter.to && dateFilter.from > dateFilter.to) {
      errors.push({
        field: 'date_range',
        message: 'The "from" date must be before the "to" date'
      });
    }
  }

  return { dateFilter, errors };
}

/**
 * Parse a date string (ISO 8601 or Unix timestamp)
 */
function parseDate(dateStr: string): Date | null {
  // Try Unix timestamp (seconds)
  const timestamp = parseInt(dateStr, 10);
  if (!isNaN(timestamp) && timestamp > 0) {
    // Detect if it's in seconds or milliseconds
    const date = timestamp > 10000000000
      ? new Date(timestamp) // milliseconds
      : new Date(timestamp * 1000); // seconds

    if (!isNaN(date.getTime())) {
      return date;
    }
  }

  // Try ISO 8601 string
  const date = new Date(dateStr);
  if (!isNaN(date.getTime())) {
    return date;
  }

  return null;
}

/**
 * Build MongoDB filter object from parsed parameters
 */
export function buildMongoFilter(filters: ClipFilters): any {
  const mongoFilter: any = {
    'processor.event': true // Base filter
  };

  // Device filter
  if (filters.device) {
    mongoFilter.device = filters.device;
  }

  // Date filter
  if (filters.dateFilter) {
    mongoFilter.created_at = {};

    if (filters.dateFilter.from) {
      mongoFilter.created_at.$gte = filters.dateFilter.from.toISOString();
    }

    if (filters.dateFilter.to) {
      mongoFilter.created_at.$lte = filters.dateFilter.to.toISOString();
    }
  }

  // Detected class filter
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

/**
 * Validate detected class parameter
 */
export function validateDetectedClass(classParam: string | null): {
  detectedClass?: string;
  errors: ClipFilterValidationError[];
} {
  const errors: ClipFilterValidationError[] = [];

  if (!classParam) {
    return { errors };
  }

  // List of valid COCO classes (80 classes)
  const validClasses = [
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

  const normalizedClass = classParam.toLowerCase().trim();

  if (!validClasses.includes(normalizedClass)) {
    errors.push({
      field: 'detectedClass',
      message: `Invalid class name. Must be one of the 80 COCO classes (e.g., person, car, dog)`
    });
    return { errors };
  }

  return { detectedClass: normalizedClass, errors };
}

/**
 * Parse all filters from URLSearchParams
 */
export function parseAllFilters(searchParams: URLSearchParams): {
  filters: ClipFilters;
  errors: ClipFilterValidationError[];
} {
  const allErrors: ClipFilterValidationError[] = [];

  // Parse pagination
  const { page, limit, errors: paginationErrors } = parsePaginationParams(searchParams);
  allErrors.push(...paginationErrors);

  // Parse date filters
  const { dateFilter, errors: dateErrors } = parseDateFilters(searchParams);
  allErrors.push(...dateErrors);

  // Parse device filter (simple string, no validation needed)
  const device = searchParams.get('device') || undefined;

  // Parse detected class filter
  const { detectedClass, errors: classErrors } = validateDetectedClass(
    searchParams.get('detectedClass')
  );
  allErrors.push(...classErrors);

  const filters: ClipFilters = {
    page,
    limit,
    device,
    detectedClass,
    dateFilter
  };

  return { filters, errors: allErrors };
}
