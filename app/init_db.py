# create_tables.py
from app.database import engine
from app.models import Feedback

print("🛠️ Creating tables...")
Feedback.__table__.create(bind=engine, checkfirst=True)
print("✅ Done.")
