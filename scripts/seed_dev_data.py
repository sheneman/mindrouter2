#!/usr/bin/env python3
############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# seed_dev_data.py: Seed database with development test data
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Seed development data for MindRouter2."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.db.session import get_async_db_context
from backend.app.db import crud
from backend.app.db.models import UserRole
from backend.app.security import hash_password, generate_api_key
from backend.app.settings import get_settings


async def seed_users():
    """Create default users for development."""
    settings = get_settings()

    users_data = [
        {
            "username": "admin",
            "email": "admin@mindrouter.local",
            "password": "admin123",
            "role": UserRole.ADMIN,
            "full_name": "Administrator",
        },
        {
            "username": "faculty1",
            "email": "faculty1@mindrouter.local",
            "password": "faculty123",
            "role": UserRole.FACULTY,
            "full_name": "Faculty User",
        },
        {
            "username": "staff1",
            "email": "staff1@mindrouter.local",
            "password": "staff123",
            "role": UserRole.STAFF,
            "full_name": "Staff User",
        },
        {
            "username": "student1",
            "email": "student1@mindrouter.local",
            "password": "student123",
            "role": UserRole.STUDENT,
            "full_name": "Student User",
        },
    ]

    async with get_async_db_context() as db:
        for user_data in users_data:
            # Check if user exists
            existing = await crud.get_user_by_username(db, user_data["username"])
            if existing:
                print(f"User {user_data['username']} already exists, skipping...")
                continue

            # Create user
            user = await crud.create_user(
                db=db,
                username=user_data["username"],
                email=user_data["email"],
                password_hash=hash_password(user_data["password"]),
                role=user_data["role"],
                full_name=user_data["full_name"],
            )
            print(f"Created user: {user.username} ({user.role.value})")

            # Create quota
            quota_defaults = settings.get_quota_defaults(user.role.value)
            await crud.create_quota(
                db=db,
                user_id=user.id,
                token_budget=quota_defaults["token_budget"],
                rpm_limit=quota_defaults["rpm"],
                max_concurrent=quota_defaults["max_concurrent"],
            )
            print(f"  Created quota: {quota_defaults['token_budget']} tokens")

            # Create an API key for each user
            full_key, key_hash, key_prefix = generate_api_key()
            await crud.create_api_key(
                db=db,
                user_id=user.id,
                key_hash=key_hash,
                key_prefix=key_prefix,
                name="Default Key",
            )
            print(f"  Created API key: {key_prefix}...")
            print(f"  FULL KEY (save this!): {full_key}")
            print()


async def main():
    """Main entry point."""
    print("=" * 60)
    print("MindRouter2 Development Data Seeder")
    print("=" * 60)
    print()

    print("Creating users...")
    await seed_users()

    print()
    print("=" * 60)
    print("Seeding complete!")
    print()
    print("Default credentials:")
    print("  admin / admin123")
    print("  faculty1 / faculty123")
    print("  staff1 / staff123")
    print("  student1 / student123")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
