#!/usr/bin/env python3
"""
Script to generate thumbnails for existing videos in B2
"""
import cv2
import os
import sys
import logging
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# Configuration
B2_KEY_ID = os.environ.get("B2_KEY_ID", "005a7351082aa2d0000000001")
B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY", "K005HOQbGe1cEaos7n3PSkB9KvdIhao")
B2_BUCKET_NAME = "visionflow-v1"
TEMP_DIR = "/tmp/visionflow_thumbnails"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_b2():
    """Initialize B2 API connection"""
    try:
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        logger.info(f"✅ Connected to B2 bucket: {B2_BUCKET_NAME}")
        return b2_api, bucket
    except Exception as e:
        logger.error(f"❌ Failed to initialize B2 API: {e}")
        sys.exit(1)

def download_video(bucket, frame_id, output_path):
    """Download video from B2"""
    try:
        video_filename = f"videos/{frame_id}.mp4"
        bucket.download_file_by_name(video_filename).save_to(output_path)
        logger.info(f"📥 Downloaded: {video_filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {frame_id}: {e}")
        return False

def generate_thumbnail(video_path, thumbnail_path):
    """Generate thumbnail from video"""
    try:
        cap = cv2.VideoCapture(video_path)

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            logger.error(f"❌ Could not read frame from {video_path}")
            cap.release()
            return False

        # Resize to 320x180 (16:9)
        thumbnail = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)

        # Save as JPEG with quality 85
        cv2.imwrite(thumbnail_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])

        cap.release()
        logger.info(f"🖼️  Generated thumbnail: {os.path.basename(thumbnail_path)}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to generate thumbnail: {e}")
        return False

def upload_thumbnail(bucket, thumbnail_path, frame_id):
    """Upload thumbnail to B2"""
    try:
        bucket_filename = f"thumbnails/{frame_id}.jpg"
        bucket.upload_local_file(
            local_file=thumbnail_path,
            file_name=bucket_filename
        )
        logger.info(f"☁️  Uploaded: {bucket_filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload thumbnail {frame_id}: {e}")
        return False

def process_video(bucket, frame_id):
    """Process a single video: download, generate thumbnail, upload"""
    os.makedirs(TEMP_DIR, exist_ok=True)

    video_path = os.path.join(TEMP_DIR, f"{frame_id}.mp4")
    thumbnail_path = os.path.join(TEMP_DIR, f"{frame_id}.jpg")

    try:
        # Download video
        if not download_video(bucket, frame_id, video_path):
            return False

        # Generate thumbnail
        if not generate_thumbnail(video_path, thumbnail_path):
            return False

        # Upload thumbnail
        if not upload_thumbnail(bucket, thumbnail_path, frame_id):
            return False

        return True
    finally:
        # Cleanup
        for path in [video_path, thumbnail_path]:
            if os.path.exists(path):
                os.remove(path)

def list_videos_without_thumbnails(bucket):
    """List all videos that don't have thumbnails"""
    logger.info("🔍 Finding videos without thumbnails...")

    # Get all video files
    videos = set()
    for file_info, _ in bucket.ls(folder_to_list="videos/"):
        filename = file_info.file_name
        if filename.endswith('.mp4'):
            frame_id = os.path.splitext(os.path.basename(filename))[0]
            videos.add(frame_id)

    logger.info(f"📹 Found {len(videos)} videos")

    # Get all thumbnails
    thumbnails = set()
    for file_info, _ in bucket.ls(folder_to_list="thumbnails/"):
        filename = file_info.file_name
        if filename.endswith('.jpg'):
            frame_id = os.path.splitext(os.path.basename(filename))[0]
            thumbnails.add(frame_id)

    logger.info(f"🖼️  Found {len(thumbnails)} existing thumbnails")

    # Find videos without thumbnails
    missing = videos - thumbnails
    logger.info(f"⚠️  {len(missing)} videos need thumbnails")

    return list(missing)

def main():
    logger.info("🚀 Thumbnail Generator for VisionFlow v2")
    logger.info("=" * 60)

    # Initialize B2
    b2_api, bucket = initialize_b2()

    # Find videos without thumbnails
    frame_ids = list_videos_without_thumbnails(bucket)

    if not frame_ids:
        logger.info("✅ All videos already have thumbnails!")
        return

    # Process each video
    logger.info(f"\n📊 Processing {len(frame_ids)} videos...")
    logger.info("=" * 60)

    success = 0
    failed = 0

    for i, frame_id in enumerate(frame_ids, 1):
        logger.info(f"\n[{i}/{len(frame_ids)}] Processing: {frame_id}")

        if process_video(bucket, frame_id):
            success += 1
        else:
            failed += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Successfully processed: {success}")
    logger.info(f"❌ Failed: {failed}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
