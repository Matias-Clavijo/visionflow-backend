/**
 * Next.js API Route: GET /api/clips
 * Event Clips API with Pagination and Filtering
 *
 * Place this file at: src/app/api/clips/route.ts in your Next.js project
 */

import { NextRequest, NextResponse } from 'next/server';
import clientPromise from '@/lib/mongodb';
import {
  parseAllFilters,
  buildMongoFilter,
  calculatePagination
} from '@/lib/clipFilters'; // Adjust path as needed
import type {
  ClipsAPIResponse,
  ClipsAPIErrorResponse
} from '@/types/clips'; // Adjust path as needed

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  console.log('🎬 [Clips API] Starting request...');

  try {
    const searchParams = request.nextUrl.searchParams;

    // Parse and validate all filters
    const { filters, errors } = parseAllFilters(searchParams);

    // If validation errors, return 400
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

    console.log(`📊 [Clips API] Filters:`, {
      page: filters.page,
      limit: filters.limit,
      device: filters.device || 'all',
      detectedClass: filters.detectedClass || 'all',
      dateFilter: filters.dateFilter ? {
        from: filters.dateFilter.from?.toISOString(),
        to: filters.dateFilter.to?.toISOString()
      } : 'none'
    });

    // Connect to MongoDB
    const client = await clientPromise;
    const db = client.db('visionflow');
    const eventsCollection = db.collection('events');

    console.log('✅ [Clips API] MongoDB connection established');

    // Build MongoDB filter
    const mongoFilter = buildMongoFilter(filters);

    console.log('🔍 [Clips API] MongoDB filter:', JSON.stringify(mongoFilter));

    // Calculate skip for pagination
    const skip = (filters.page - 1) * filters.limit;

    // Execute queries in parallel for performance
    const [events, totalCount] = await Promise.all([
      eventsCollection
        .find(mongoFilter)
        .sort({ created_at: -1 })
        .skip(skip)
        .limit(filters.limit)
        .toArray(),
      eventsCollection.countDocuments(mongoFilter)
    ]);

    console.log(`📦 [Clips API] Found ${events.length} events from MongoDB (total: ${totalCount})`);

    if (events.length > 0) {
      console.log('🔍 [Clips API] First event sample:', {
        frame_id: events[0].frame_id,
        device: events[0].device,
        tags: events[0].processor?.tags?.length || 0,
        created_at: events[0].created_at
      });
    }

    // Map MongoDB documents to EventClip interface with proxy URLs
    const clips = events.map((event: any) => {
      // Use our API proxy endpoint to stream videos
      const videoUrl = `/api/clips/stream/${event.frame_id}`;

      return {
        _id: event._id.toString(),
        frame_id: event.frame_id,
        device: event.device,
        timestamp: event.timestamp,
        quality_reduction: event.quality_reduction,
        processor: event.processor,
        created_at: event.created_at,
        video_url: videoUrl,
      };
    });

    // Calculate pagination metadata
    const pagination = calculatePagination(filters.page, filters.limit, totalCount);

    console.log(`📊 [Clips API] Pagination:`, pagination);

    // Build response with applied filters info
    const appliedFilters: any = {};
    if (filters.device) {
      appliedFilters.device = filters.device;
    }
    if (filters.detectedClass) {
      appliedFilters.detectedClass = filters.detectedClass;
    }
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
    console.log(`✅ [Clips API] Request completed successfully in ${duration}ms`);

    return NextResponse.json(response);
  } catch (error) {
    console.error('❌ [Clips API] Error fetching clips:', error);

    const errorResponse: ClipsAPIErrorResponse = {
      success: false,
      error: 'Internal Server Error',
      message: error instanceof Error ? error.message : 'Unknown error occurred'
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
