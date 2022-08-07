from eeg_positions import get_elec_coords, plot_coords

coords = get_elec_coords(system="1005", dim="3d")

x = coords.get('x')
y = coords.get('y')
z = coords.get('z')

with open(r"/home/anne/Masterarbeit/masterarbeit/data/electrodes_1005.txt", 'w') as fp:
    for i in range(len(x)):
        fp.write("%s %s %s\n" % (x[i], y[i], z[i]))


coords = get_elec_coords(system="1010", dim="3d")

x = coords.get('x')
y = coords.get('y')
z = coords.get('z')

with open(r"/home/anne/Masterarbeit/masterarbeit/data/electrodes_1010.txt", 'w') as fp:
    for i in range(len(x)):
        fp.write("%s %s %s\n" % (x[i], y[i], z[i]))


coords = get_elec_coords(system="1020", dim="3d")

x = coords.get('x')
y = coords.get('y')
z = coords.get('z')

with open(r"/home/anne/Masterarbeit/masterarbeit/data/electrodes_1020.txt", 'w') as fp:
    for i in range(len(x)):
        fp.write("%s %s %s\n" % (x[i], y[i], z[i]))


