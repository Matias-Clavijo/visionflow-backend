/**
 * TypeScript types for Event Clips API with Pagination
 */

export interface EventClip {
  _id: string;
  frame_id: string;
  device: string;
  timestamp: number;
  quality_reduction: {
    original_size_bytes: number;
    reduced_size_bytes: number;
    compression_ratio: number;
    frame_skip: number;
    quality_factor: number;
    target_resolution: string;
    jpeg_quality: number;
  };
  processor: {
    count: number;
    cached: boolean;
    tags: Array<{
      class_id: number;
      class_name: string;
      confidence: number;
      bbox: {
        x: number;
        y: number;
        width: number;
        height: number;
        center_x: number;
        center_y: number;
      };
    }>;
    event: boolean;
    frame_info: {
      width: number;
      height: number;
      channels: number;
    };
    model_info: {
      model_path: string;
      confidence_threshold: number;
      nms_threshold: number;
    };
    performance: {
      processing_time_ms: {
        total: number;
        preprocess: number;
        inference: number;
        postprocess: number;
      };
    };
  };
  created_at: string;
  video_url?: string;
}

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

/**
 * Query parameters for GET /api/clips
 */
export interface ClipsQueryParams {
  // Pagination
  page?: number;
  limit?: number;

  // Filters
  device?: string;
  detectedClass?: string;

  // Date filters
  from?: string; // ISO 8601 or Unix timestamp
  to?: string; // ISO 8601 or Unix timestamp
  date?: string; // Exact date (ISO 8601 or Unix timestamp)
}
