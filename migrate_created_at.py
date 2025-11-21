#!/usr/bin/env python3
"""
Migration script to convert created_at from ISO string to datetime object
Run this once to fix existing clips in MongoDB
"""

from pymongo import MongoClient
from datetime import datetime
import sys

def migrate_created_at():
    """Convert all created_at fields from ISO strings to datetime objects"""
    try:
        # Connect to MongoDB (using Atlas cloud connection)
        mongo_uri = 'mongodb+srv://tesis:ucu2025tesis@visionflow.92xlyhu.mongodb.net/'
        client = MongoClient(mongo_uri)
        db = client['visionflow']
        collection = db['events']

        print("🔍 Checking for clips with string created_at...")

        # Find all documents where created_at is a string
        string_dates = collection.find({
            'created_at': {'$type': 'string'}
        })

        count = 0
        updated = 0
        errors = 0

        for doc in string_dates:
            count += 1
            try:
                # Parse ISO string to datetime
                created_at_str = doc['created_at']
                created_at_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))

                # Update document
                result = collection.update_one(
                    {'_id': doc['_id']},
                    {'$set': {'created_at': created_at_dt}}
                )

                if result.modified_count > 0:
                    updated += 1
                    print(f"✅ Updated {doc.get('frame_id', 'unknown')}: {created_at_str} -> {created_at_dt}")

            except Exception as e:
                errors += 1
                print(f"❌ Error updating {doc.get('frame_id', 'unknown')}: {e}")

        print("\n" + "=" * 60)
        print(f"📊 Migration complete:")
        print(f"   Found: {count} clips with string dates")
        print(f"   Updated: {updated} clips")
        print(f"   Errors: {errors} clips")
        print("=" * 60)

        # Verify the migration
        remaining_strings = collection.count_documents({
            'created_at': {'$type': 'string'}
        })

        if remaining_strings > 0:
            print(f"⚠️  Warning: {remaining_strings} clips still have string dates")
        else:
            print("✅ All clips now have datetime objects for created_at")

        client.close()
        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("MongoDB created_at Migration Script")
    print("Converting ISO strings to datetime objects")
    print("=" * 60 + "\n")

    success = migrate_created_at()
    sys.exit(0 if success else 1)
