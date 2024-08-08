# Dynamic routing code
import numpy as np

def dynamic_routing(data):
    # Implement dynamic routing
    route = []
    for point in data:
        if point[2] > -1.5:
            route.append(point)
    return route
