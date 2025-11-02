import mysql.connector

conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    database="student-management"
)

cursor = conn.cursor()


sql = "INSERT INTO student (code, name, age) VALUES (%s, %s, %s)"
val = ("sv07", "Trần Duy Thanh", 45)
cursor.execute(sql, val)
print(cursor.rowcount, "record inserted")

sql = "INSERT INTO student (code, name, age) VALUES (%s, %s, %s)"
val = [
    ("sv08", "Trần Quyết Chiến", 19),
    ("sv09", "Hồ Thắng", 22),
    ("sv10", "Hoàng Hà", 25),
]
cursor.executemany(sql, val)
print(cursor.rowcount, "records inserted")

conn.commit()

cursor.close()
conn.close()
