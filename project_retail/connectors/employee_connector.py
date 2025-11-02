from project_retail.connectors.connector import Connector
from project_retail.models.employee import Employee


class EmployeeConnector(Connector):
    def login(self,uid,pwd):
        sql = "SELECT * FROM employee " \
              "where Email=%s and Password=%s"
        val = (uid, pwd)
        dataset=self.fetchone(sql,val)
        emp = None
        if dataset != None:
            emp = Employee(dataset[0], dataset[1], dataset[2], dataset[3],
                           dataset[4], dataset[5])
        return emp
    def get_list_employee(self):
        sql = "SELECT * FROM employee "
        datasets=self.fetchall(sql,None)
        employees=[]
        if datasets!=None:
            for dataset in datasets:
                emp = Employee(dataset[0], dataset[1], dataset[2], dataset[3],
                               dataset[4], dataset[5])
                employees.append(emp)
        return employees
    def insert_employee(self,emp):
        sql="INSERT "\
        " INTO "\
        " `employee` "\
        "(" \
            " `Name`, " \
            " `Email`, " \
            " `Phone`, " \
            " `Password`, " \
            " `IsDeleted`) "\
        " VALUES "\
        " (%s,%s,%s,%s,%s);"
        val=(emp.Name,emp.Email,emp.Phone,emp.Password,emp.IsDeleted)
        result=self.savedata(sql,val)
        return result
    def update_employee(self,emp):
        sql="UPDATE "\
        " `employee` "\
        " SET "\
        " `Name` = %s,"\
        " `Email`= %s,"\
        " `Phone` = %s,"\
        " `Password` = %s,"\
        " `IsDeleted` = %s"\
        " WHERE "\
        " `ID` = %s"
        val = (emp.Name, emp.Email, emp.Phone, emp.Password, emp.IsDeleted,emp.ID)
        result = self.savedata(sql, val)
        return result
    def get_detail(self,id):
        sql = "SELECT * FROM employee " \
              "where ID=%s"
        val = (id,)
        dataset = self.fetchone(sql, val)
        emp = None
        if dataset != None:
            emp = Employee(dataset[0], dataset[1], dataset[2], dataset[3],
                           dataset[4], dataset[5])
        return emp