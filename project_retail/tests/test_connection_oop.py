from project_retail.connectors.employee_connector import EmployeeConnector

ec=EmployeeConnector()
ec.connect()
em=ec.login("bin@gmail.com",123)
if em==None:
    print("Login failed")
else:
    print(em)

print("Test - All Employees:")
employees=ec.get_list_employee()
for emp in employees:
    print(emp)

print("----------ID=4-------------")
id=4
emp=ec.get_detail(id)
if emp!=None:
    print("FOUND ID=",id)
    print(emp)
else:
    print("NOT FOUND ID=",id)