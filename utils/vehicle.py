global VEHICLES # List of vehicles
VEHICLES = {}

class Vehicle:
    def __init__(self, id, plate_number, location):
        self.id = id
        self.plate_number = plate_number
        self.location = location
        self.illegal = False

    def getId(self):
        return self.id

    def setId(self, id):
        self.id = id

    def getplate_number(self):
        return self.plate_number

    def setplate_number(self, plate_number):
        self.plate_number = plate_number

    def getLocation(self):
        return self.location

    def setLocation(self, location):
        self.location = location

    def isIllegal(self):
        return self.illegal

    def setIllegal(self, illegal):
        self.illegal = illegal
