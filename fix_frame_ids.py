#!/usr/bin/env python3
"""
Migration script to add 'clip_' prefix to frame_id for frontend clips
This ensures MongoDB frame_ids match the actual filenames in B2
"""

from pymongo import MongoClient
import sys

def fix_frame_ids():
    """Add 'clip_' prefix to frame_ids that start with 'frontend_' but don't have 'clip_'"""
    try:
        # Connect to MongoDB (using Atlas cloud connection)
        mongo_uri = 'mongodb+srv://tesis:ucu2025tesis@visionflow.92xlyhu.mongodb.net/'
        client = MongoClient(mongo_uri)
        db = client['visionflow']
        collection = db['events']

        print("🔍 Checking for frame_ids that need 'clip_' prefix...")

        # Find all documents where frame_id starts with 'frontend_' but doesn't contain 'clip_'
        docs = collection.find({
            'frame_id': {
                '$regex': '^frontend_',
                '$not': {'$regex': 'clip_'}
            }
        })

        count = 0
        updated = 0
        errors = 0

        for doc in docs:
            count += 1
            try:
                old_frame_id = doc['frame_id']
                new_frame_id = f"clip_{old_frame_id}"

                # Update document
                result = collection.update_one(
                    {'_id': doc['_id']},
                    {'$set': {'frame_id': new_frame_id}}
                )

                if result.modified_count > 0:
                    updated += 1
                    print(f"✅ Updated: {old_frame_id} -> {new_frame_id}")

            except Exception as e:
                errors += 1
                print(f"❌ Error updating {doc.get('frame_id', 'unknown')}: {e}")

        print("\n" + "=" * 60)
        print(f"📊 Migration complete:")
        print(f"   Found: {count} clips needing update")
        print(f"   Updated: {updated} clips")
        print(f"   Errors: {errors} clips")
        print("=" * 60)

        # Verify the migration
        remaining = collection.count_documents({
            'frame_id': {
                '$regex': '^frontend_',
                '$not': {'$regex': 'clip_'}
            }
        })

        if remaining > 0:
            print(f"⚠️  Warning: {remaining} clips still need 'clip_' prefix")
        else:
            print("✅ All frontend clips now have 'clip_' prefix in frame_id")

        client.close()
        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("MongoDB frame_id Fix Script")
    print("Adding 'clip_' prefix to frontend frame_ids")
    print("=" * 60 + "\n")

    success = fix_frame_ids()
    sys.exit(0 if success else 1)
