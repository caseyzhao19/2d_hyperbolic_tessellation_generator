class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, other=None):
        if not other:
            other = Point(0,0)
        return ((self.x-other.x)**2+(self.y-other.y)**2)**(1/2)

    def inverse(self, c, r):
        d = self.dist(c)
        if d == 0:
            print("origin has no inverse")
            return None
        return Point(c.x+(r**2)*(self.x-c.x)/(d**2), c.y+(r**2)*(self.y-c.y)/(d**2))

    def midpoint(self, other):
        return Point((self.x+other.x)/2, (self.y+other.y)/2)

    def slope(self, other):
        if other.x == self.x:
            return 9999999999999
        return (other.y-self.y)/(other.x-self.x)

    def __eq__(self, other):
        return abs(self.x-other.x)<0.000001 and abs(self.y-other.y)<0.000001

    def isCollinear(self, o1, o2):
        if self.__eq__(o1) or self.__eq__(o2) or o1.__eq__(o2):
            return True
        return abs(self.slope(o1)-self.slope(o2)) < 0.0000001