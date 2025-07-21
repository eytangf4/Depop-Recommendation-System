#!/usr/bin/env python3

import sqlite3
from werkzeug.security import generate_password_hash

# Connect to database
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Update password for user eytangf
password_hash = generate_password_hash("Dismagniv4!")
cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (password_hash, "eytangf"))
conn.commit()

print(f"✅ Updated password for user 'eytangf'")
print(f"Rows affected: {cursor.rowcount}")

conn.close()
