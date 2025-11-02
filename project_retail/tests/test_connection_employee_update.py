from project_retail.connectors.employee_connector import EmployeeConnector
from project_retail.models.employee import Employee

ec=EmployeeConnector()
ec.connect()
emp=ec.get_detail(4)
emp.Name="Trần Thị Chín"
emp.Email="999@gmail.com"
result=ec.update_employee(emp)
if result>0:
    print("UPDATE OK")
else:
    print("UPDATE FAILED")