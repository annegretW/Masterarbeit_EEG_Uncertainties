import meshio 

def get_midpoint(a,b,c,d):
    x = (a[0]+b[0]+c[0]+d[0])/4
    y = (a[1]+b[1]+c[1]+d[1])/4
    z = (a[2]+b[2]+c[2]+d[2])/4
    return x,y,z


mesh = meshio.read("/home/anne/Masterarbeit/masterarbeit/data/tet_mesh.msh")

cells = mesh.cells[0].data
points = mesh.points

midpoints = []

with open(r"/home/anne/Masterarbeit/masterarbeit/data/midpoints.txt", 'w') as fp:
    for cell in cells:
        #midpoints.append(get_midpoint(points[cell[0]],points[cell[1]],points[cell[2]],points[cell[3]]))
        fp.write("%s %s %s\n" % get_midpoint(points[cell[0]],points[cell[1]],points[cell[2]],points[cell[3]]))
