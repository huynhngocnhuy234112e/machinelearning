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
cursor = conn.cursor()
sql="update student set name='Hoàng Lão Tà' where Code='sv09'"
cursor.execute(sql)

conn.commit()

print(cursor.rowcount," record(s) affected")
cursor = conn.cursor()
sql="update student set name=%s where Code=%s"
val=('Hoàng Lão Tà','sv09')

cursor.execute(sql,val)

conn.commit()

print(cursor.rowcount," record(s) affected")