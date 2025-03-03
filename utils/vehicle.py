global VEHICLES # List of vehicles
VEHICLES = {}

class Vehicle:
    def __init__(self, id, plate, location):
        self.id = id
        self.plate = plate
        self.location = location
        self.illegal = False

    def getId(self):
        return self.id

    def setId(self, id):
        self.id = id

    def getPlate(self):
        return self.plate

    def setPlate(self, plate):
        self.plate = plate

    def getLocation(self):
        return self.location

    def setLocation(self, location):
        self.location = location

    def isIllegal(self):
        return self.illegal

    def setIllegal(self, illegal):
        self.illegal = illegal
