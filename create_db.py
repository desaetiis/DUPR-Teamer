import sqlite3

def create_database():
    conn = sqlite3.connect('players.db')
    cursor = conn.cursor()

    # Create players table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        gender TEXT NOT NULL,
        dupr_rating REAL NOT NULL
    )
    ''')

    # Insert sample data
    sample_data = [
        ("Mike", "Male", 4.722),
        ("Dan", "Male", 4.478),
        ("Kim", "Female", 4.963),
        ("Ashley", "Female", 4.914),
        ("Jane", "Female", 4.920),
        ("Tim", "Male", 4.622),
        ("Kam", "Female", 4.434),
        ("Alex", "Male", 5.160)
    ]

    cursor.executemany('''
    INSERT INTO players (name, gender, dupr_rating)
    VALUES (?, ?, ?)
    ''', sample_data)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()