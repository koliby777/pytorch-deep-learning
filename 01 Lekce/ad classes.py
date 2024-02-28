class Parent:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

print(" Parent:")
nic = Parent("Rodic", "Velky") 
nic.printname()

class Child(Parent):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    # By calling super().__init__(), you can execute the __init__ method of the parent class as if it were defined in the current class
    self.graduationyear = year

  def welcome(self):
    print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)

print("\n Child:")
x = Child("Dite", "Male", 2024)
x.printname() # tato fce je zdedena
x.welcome()
