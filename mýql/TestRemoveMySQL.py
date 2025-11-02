import mysql.connector

server = "localhost"
port = 3306
database = "student-management"
username = "root"

conn = mysql.connector.connect(
    host=server,
    port=port,
    database=database,
    user=username,)
conn = mysql.connector.connect(
                host=server,
                port=port,
                database=database,
                user=username,
                )
cursor = conn.cursor()
sql="DELETE from student where ID=14"
cursor.execute(sql)

conn.commit()

print(cursor.rowcount," record(s) affected")
conn = mysql.connector.connect(
                host=server,
                port=port,
                database=database,
                user=username,
               )
cursor = conn.cursor()
sql = "DELETE from student where ID=%s"
val = (13,)

cursor.execute(sql, val)

conn.commit()

print(cursor.rowcount," record(s) affected")
