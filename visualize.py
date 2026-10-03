import os
import matplotlib.pyplot as plt

def plot_orbit_3d(r_ECI_ARRAY, save_path, title='Orbit_v1'):
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw={'projection':'3d'}, facecolor = 'black') #huabu
    ax.set_facecolor('black')

    ax.plot(r_ECI_ARRAY[:,0], r_ECI_ARRAY[:,1], r_ECI_ARRAY[:,2], color = 'cyan', linewidth = 1.5, label = title)
    ax.scatter([0], [0], [0], s = 300, c = 'steelblue', label = 'Earth')

    ax.set_box_aspect([1,1,1])
    ax.legend(labelcolor = 'white')

    #save_path = '/home/osyark/orbit_V1/orbit.png'
    plt.savefig(save_path, dpi=150)
    print("图片保存到:", os.path.abspath(save_path))
    plt.close()

"""def plot_time_series(t_array, r_array, v_array):
    ..."""