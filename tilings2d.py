import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

delta = 0.000001

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, other=None):
        if not other:
            other = Point(0,0)
        return ((self.x-other.x)**2+(self.y-other.y)**2)**(1/2)

    # if circle is infinite radius, is the same as line reflection
    # indicate by r = -1 and c = (a,b) is two points on the line
    def inverse(self, c=None, r=1):
        if c is None:
            c = Point(0,0)
        if r == -1:
            (j,k) = c
            m = j.slope(k)
            if m is None:
                return Point(2*j.x-self.x, self.y)
            t = -2*(m*self.x-self.y+j.y-m*j.x)/(m**2+1)
            return Point(m*t+self.x,-t+self.y)
        d = self.dist(c)
        if d == 0:
            print("origin has no inverse")
            return None
        return Point(c.x+(r**2)*(self.x-c.x)/(d**2), c.y+(r**2)*(self.y-c.y)/(d**2))

    def midpoint(self, other):
        return Point((self.x+other.x)/2, (self.y+other.y)/2)

    def slope(self, other):
        if abs(other.x - self.x) < delta:
            return None
        return (other.y-self.y)/(other.x-self.x)

    # ah everything is very ugly i am sorry
    def __eq__(self, other):
        return abs(self.x-other.x) < delta and abs(self.y-other.y) < delta

    def __hash__(self):
        return round(1/delta * ((self.x + self.y) * (self.x + self.y + 1)/2) + self.y)

    def isCollinear(self, o1, o2):
        if self.__eq__(o1) or self.__eq__(o2) or o1.__eq__(o2):
            return True
        s1 = self.slope(o1)
        s2 = self.slope(o2)
        if s1 is None or s2 is None:
            return s1 is None and s2 is None
        return abs(s1-s2) < delta

figure, axes = plt.subplots()

def intersect(a,b,c,d):
    px = ((a.x*b.y-a.y*b.x)*(c.x-d.x)-(a.x-b.x)*(c.x*d.y-c.y*d.x))/(
                (a.x-b.x)*(c.y-d.y)-(a.y-b.y)*(c.x-d.x))
    py = ((a.x*b.y-a.y*b.x)*(c.y-d.y)-(a.y-b.y)*(c.x*d.y-c.y*d.x))/(
                (a.x-b.x)*(c.y-d.y)-(a.y-b.y)*(c.x-d.x))
    return Point(px, py)

def circleIntersect(c1, r1, c2, r2):
    # stolen also :(
    # i was having weird issues so i gave up
    d = c1.dist(c2)
    a = (r1**2-r2**2+d**2)/(2*d)
    h = (r1**2-a**2)**(1/2)

    x2 = c1.x+a*(c2.x-c1.x)/d   
    y2 = c1.y+a*(c2.y-c1.y)/d

    x3 = x2+h*(c2.y-c1.y)/d     
    y3 = y2-h*(c2.x-c1.x)/d
    x4 = x2-h*(c2.y-c1.y)/d
    y4 = y2+h*(c2.x-c1.x)/d
    
    return Point(x3, y3), Point(x4, y4)

def perpBisect(a, b, plot=False):
    c1, r1 = hyperCircle(a,b)
    c2, r2 = hyperCircle(b,a)
    p1, p2 = circleIntersect(c1, r1, c2, r2)
    return hyperLine(p1, p2, plot, False)

def hyperCircle(c, pt, plot=False):
    if c == Point(0, 0):
        C = c
        R = C.dist(pt)
    elif c.isCollinear(pt, Point(0,0)):
        ci = c.inverse()
        c2 = c.midpoint(ci)
        ai = pt.inverse(c2, c2.dist(c))
        C = pt.midpoint(ai)
        R = C.dist(ai)
    else:
        ct, rt = hyperLine(c, pt)
        t = tangent(ct, pt)
        C = intersect(t[0],t[1],Point(0,0), c)
        R = C.dist(pt)
    if plot:
        plt.scatter(c.x, c.y)
        plt.scatter(pt.x,pt.y)
        axes.add_patch(plt.Circle((C.x, C.y), R, fill=False))
    return C, R

def hyperLine(a, b, plot=False, arc=True):
    if a.dist() == 0 or b.dist() == 0 or a.isCollinear(b, Point(0,0)):
        if plot: plt.plot([a.x,b.x],[a.y,b.y], color=(0,0,0), linewidth=0.9)
        return (a,b), -1
    ai = a.inverse()
    if not ai:
        ai = b.inverse()
    return threePointCircle(a, b, ai, plot, arc)

def threePointCircle(a, b, c, plot=False, arc=True):
    # stolen from stackexchange, is very clevery
    z1 = complex(a.x, a.y)
    z2 = complex(b.x, b.y)
    z3 = complex(c.x, c.y)
    w = (z3 - z1) / (z2 - z1)
    if w.imag == 0: print('bad')
    c = (z2 - z1) * (w - abs(w) ** 2) / (2j * w.imag) + z1
    r = abs(z1 - c)
    c = Point(c.real, c.imag)
    if plot and not arc:
        axes.add_patch(plt.Circle((c.x,c.y), r, fill=False))
    elif plot and arc:
        theta1 = threeSixtyArctan(c, a)
        theta2 = threeSixtyArctan(c, b)
        small = theta1 if (theta2 - theta1) % 360 < 180 else theta2
        big = theta2 if (theta2 - theta1) % 360 < 180 else theta1
        arc = ptc.Arc((c.x, c.y), 2*r, 2*r, 0, small, big)
        axes.add_patch(arc)
    return c, r

# given a center point c of circle and point p on circle,
# find angle of p relative to c starting from rightmost point of
# circle and going counterclockwise
def threeSixtyArctan(c, p):
    s = p.slope(c)
    theta = 90
    if s is not None: theta = (np.arctan(s)*180/np.pi)
    if theta == 90:
        if p.y-c.y < 0:
            theta += 180
    elif p.x-c.x < 0:
        theta += 180
    return theta

def tangent(c, pt, plot=False, l=10):
    slope = c.slope(pt)
    if slope == 0:
        p1 = Point(pt.x, pt.y + l)
        p2 = Point(pt.x, pt.y - l)
    elif slope is None:
        p1 = Point(pt.x-l, pt.y)
        p2 = Point(pt.x+l, pt.y)
    else:
        p1 = Point(pt.x-l, pt.y+l/slope)
        p2 = Point(pt.x+l, pt.y-l/slope)
    if plot: plt.plot([p1.x,p2.x],[p1.y,p2.y], color=(0,0,0), linewidth=0.9)
    return p1, p2

def populate(old_edges, new_edges, s, q):
    for e, ps in old_edges:
        c, r = e
        if 0 <= r < 0.03: continue
        pts2 = [p.inverse(c, r) for p in ps]
        for i in range(q):
            p1 = Point(pts2[i].x, pts2[i].y)
            p2 = Point(pts2[(i+1)%q].x, pts2[(i+1)%q].y)
            check = frozenset([p1,p2])
            if check not in s:
                edge = hyperLine(p1, p2, True)
                new_edges.append((edge, pts2))
            s.add(check)

class Shape:
    # center shape
    def __init__(self, p, q, c=None, r=0):
        d = ((np.tan(np.pi / 2 - np.pi / q) - np.tan(np.pi / p)) /
             (np.tan(np.pi / 2 - np.pi / q) + np.tan(np.pi / p))) ** (1 / 2)
        angles = [np.pi * 2 / q * n + (r*180/np.pi) for n in range(q)]
        pts = [Point(d * np.cos(angle), d * np.sin(angle)) for angle in angles]
        if not c or c == Point(0,0):
            pts_shift = pts
        else:
            C, R = perpBisect(Point(0,0), c)
            pts_shift = [p.inverse(C, R) for p in pts]

        edges = []
        s = set()
        for i in range(q):
            p1 = Point(pts_shift[i].x, pts_shift[i].y)
            p2 = Point(pts_shift[(i+1)%q].x, pts_shift[(i+1)%q].y)
            edges.append((hyperLine(p1, p2, True), pts_shift))
            s.add(frozenset([p1,p2]))

        for k in range(4):
            new_edges = []
            populate(edges, new_edges, s, q)
            edges = []
            populate(new_edges, edges, s, q)
p=int(sys.argv[1])
q=int(sys.argv[2])
x=float(sys.argv[3])
y=float(sys.argv[4])
unit = plt.Circle((0,0), 1, fill=0)
axes.set_aspect(1)
axis_len = 1.1
plt.xlim([-axis_len,axis_len])
plt.ylim([-axis_len,axis_len])
axes.add_patch(unit)
p1 = Point(x,y)
Shape(p,q, p1)
plt.savefig('tiling.jpg', dpi=600)
print('hi')