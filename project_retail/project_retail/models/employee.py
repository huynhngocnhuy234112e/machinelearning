class Employee:
    def __init__(self,ID=None,Name=None,Email=None,
                 Phone=None,Password=None,IsDeleted=None):
        self.ID=ID
        self.Name=Name
        self.Email=Email
        self.Phone=Phone
        self.Password=Password
        self.IsDeleted=IsDeleted
    def __str__(self):
        infor="{}\t{}\t{}\t{}".format(self.ID,self.Name,
                                      self.Email,self.Phone)
        return infor
