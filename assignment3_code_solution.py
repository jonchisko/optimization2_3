import numpy as np
import matplotlib.pyplot as plt



"""
1/(1 + ((l2-x2)/(l1-x1))^2) * (l2-x2)/(l1-x1)^2
1/(1 + ((l2-x2)/(l1-x1))^2) * (x1-l1)/(l1-x1)^2
"""

def der_arc_tan(x):
    return 1/(1+x*x)

def g_stationary(towers, x, z):
    g = list(0 for i in range(3))
    for i in range(len(towers[0])):
        numer = (towers[1][i] - x[1])
        denumer = (towers[0][i] - x[0])
        a = z[i] - np.arctan(numer/denumer)
        g[i] = a
    d = np.array(g)
    return d

def dg_stationary(towers, x):
    deg = np.zeros((len(x), len(towers[0])))
    for col in range(3):
        tower = towers[:, col]
        K = der_arc_tan((tower[1] - x[1]) / (tower[0] - x[0]))
        a = (tower[1] - x[1])
        b = (tower[0] - x[0])**2
        c = (x[0] - tower[0])
        sub1 = a/b
        sub2 = c/b
        deg[0][col] = -K * sub1
        deg[1][col] = -K * sub2
    return deg

def estimate_position(towers, z):
    xi = [np.array([2.1, 2.1])]
    zest = [np.array([0, 0, 0])]
    H = np.zeros((2, 2))
    lamda = 0.9

    plt.figure(1, figsize=(7, 7))
    fig1 = plt.gcf()
    # generate 2D grid
    x_1, x_2 = np.mgrid[-20:20:0.1, -27:20:0.1]
    plt.plot(xi[0][0], xi[0][1], '*', markersize=10, color='red')
    for col in range(len(towers[0])):
        t = towers[:, col]
        plt.plot(t[0], t[1], '+', markersize=20, color='blue')


    g = g_stationary
    dg = dg_stationary

    g_plot = 0

    for dd, zi in enumerate(z):
        x = xi[-1]
        H = lamda * H + dg(towers, x).dot(np.transpose(dg(towers, x)))
        x_new = x - np.linalg.inv(H).dot(dg(towers, x)).dot(g(towers, x, zi))
        xi.append(x_new)
        zest.append(g(towers, x_new, zi))

        plt.plot((x[0], x_new[0]), (x[1], x_new[1]), linewidth=2.0, color="black")
        plt.plot(x[0], x[1], "*", color="black", markersize=7)

        if dd == len(z)-1:
            for ind in range(0, 3):
                numer = (towers[1][ind] - x_2)
                denumer = (towers[0][ind] - x_1)
                g_plot += zi[ind] - np.arctan(numer / denumer)

        fig1.canvas.draw()
    x = xi[-1]
    plt.plot(x[0], x[1], "*", color = "green", markersize=14)
    plt.title("Pose estimation, lambda = %.2f" % (lamda))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    ddd = plt.contour(x_1, x_2, g_plot, 100)
    plt.clabel(ddd, inline=1, fontsize=8)
    fig1.canvas.draw()
    plt.savefig("pose_estimation_lambda=%s" % (str(lamda).replace(".", "_")))


    print(xi[-1])
    print(zest[-1])

    plt.show()

def g_motion(towers, x, z, t):
    g = [0, 0, 0]
    posx = x[0]
    posy = x[1]
    vx = x[2]
    vy = x[3]
    for col in range(len(towers[0])):
        tower = towers[:, col]
        g[col] = z[col] - np.sqrt((tower[0] - (posx + t*vx))**2 + (tower[1] - (posy + t*vy))**2)
    #print(np.array(g))
    return np.array(g)

def dg_motion(towers, x, t):
    deg = np.zeros((4, len(towers[0])))
    posx = x[0]
    posy = x[1]
    vx = x[2]
    vy = x[3]
    for col in range(3):
        tower = towers[:, col]
        numerator1 = posx + t*vx - tower[0]
        numerator2 = posy + t*vy - tower[1]
        denumerator = np.sqrt((tower[0] - (posx + t*vx))**2 + (tower[1] - (posy + t*vy))**2)
        deg[0][col] = -(numerator1/denumerator)
        deg[1][col] = -(numerator2/denumerator)
        deg[2][col] = -t*(numerator1/denumerator)
        deg[3][col] = -t*(numerator2/denumerator)
    #print(deg)
    return deg


def estimate_motion(towers, z):

    # generate 2D grid
    x_1, x_2 = np.mgrid[-10:27:0.1, -27:17:0.1]

    xi = [np.array([-4.01, -20.46, 0, 0])]
    zest = [np.array([0, 0, 0])]
    H = np.zeros((4, 4))
    lamda = 0.9

    threshold = 0.01

    g = g_motion
    dg = dg_motion

    all_position = [[xi[0][0], xi[0][1]]]


    time_to_plot = len(z)


    plt.figure(1, figsize=(7, 7))
    fig1 = plt.gcf()
    plt.plot(xi[0][0], xi[0][1], '*', markersize=10, color='red')
    for col in range(len(towers[0])):
        t = towers[:, col]
        plt.plot(t[0], t[1], '+', markersize=20, color='blue')

    directions = []

    for t, zi in enumerate(z):
        t+=1
        x = xi[-1]
        H = lamda * H + dg(towers, x, t).dot(np.transpose(dg(towers, x, t)))
        x_new = x - np.linalg.pinv(H).dot(dg(towers, x, t)).dot(g(towers, x, zi, t))

        prev_pos = all_position[-1]
        all_position.append([x_new[0] + t*x_new[2], x_new[1] + t*x_new[3]])
        cur_pos = all_position[-1]

        xi.append(x_new)
        zest.append(g(towers, x_new, zi, t))
        plt.plot((prev_pos[0], cur_pos[0]), (prev_pos[1], cur_pos[1]), linewidth=2.0, color="black")
        plt.plot(prev_pos[0], prev_pos[1], "*", color="black", markersize=7)

        g_plot = 0
        fig1.canvas.draw()
        if t == time_to_plot:
            for ind in range(0, 3):
                g_plot += (zi[ind]-np.sqrt((x_1-towers[0][ind])**2+(x_2-towers[1][ind])**2))**2

            print("POSITION AT:" + str(time_to_plot))
            print(all_position[-1])
            print("X, X")
            print(xi[-1][:2])
            print("V, V")
            print(xi[-1][2:])
            prev_pos = all_position[-1]

            plt.contour(x_1, x_2, g_plot, 150)

            plt.plot(prev_pos[0], prev_pos[1], "*", color="green", markersize=14)

            plt.title("Motion estimation, lambda = %.2f, t=%d" % (lamda, t))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
            ddd = plt.contour(x_1, x_2, g_plot, 70)
            plt.clabel(ddd, inline=1, fontsize=8)
            fig1.canvas.draw()

            plt.savefig("motion_estimation(t="+str(t)+",lambda="+str(lamda).replace(".", "_")+")")



    print(xi[-1])
    print(zest[-1])
    print(all_position[-1])

    plt.show()

    plt.figure(1, figsize=(7, 7))
    fig1 = plt.gcf()
    for ind in range(len(xi)-1):
        v1 = xi[ind][2:]
        v2 = xi[ind+1][2:]
        directions.append(v2-v1)
    origin = [0], [0]
    V = np.ones((len(xi), 2))
    colorT = []
    xdif = []
    ydif = []
    dir_changes = 0
    for i in range(0, len(xi)-1):
        #t = i + 1
        dif1 = xi[i+1][2] - xi[i][2]
        dfi2 = xi[i+1][3] - xi[i][3]
        dolzina = np.sqrt(dif1**2 + dfi2**2)
        print(dolzina)
        if dolzina >= threshold:
            dir_changes += 1
        xdif.append(dif1)
        ydif.append(dfi2)
        colorT.append(i)
    cmap = plt.cm.get_cmap('plasma')

    print("DIRECTION CHANGES", lamda, dir_changes)

    plt.quiver(*origin, V[:, 0], V[:, 1], color=[cmap(value) for value in colorT], scale=2, alpha=0.6)
    plt.xlim([-0.0007, 0.00076])
    plt.ylim([-0.0007, 0.00076])
    plt.savefig("Directions")
    #for direct in directions:
        #plt.plot([0, 0], [direct[0], direct[1]], linewidth=2.0, color="black")
    #plt.plot([x[2] for x in xi],[x[3] for x in xi], "*", color="black", markersize=7)
    fig1.canvas.draw()
    plt.show()

    plt.figure(1, figsize=(7, 7))
    fig1 = plt.gcf()

    plt.scatter(xdif, ydif, c=[cmap(value) for value in colorT])
    plt.title("Differences between the course vectors, lambda=%0.2f"%(lamda))
    plt.xlabel("x")
    plt.ylabel("y")
    fig1.canvas.draw()
    plt.savefig("scatterTest"+str(lamda).replace(".", "_"))
    plt.show()

if __name__ == '__main__':
    # load the data
    data = np.load('./data_position.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z'] * np.pi/180

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)




    estimate_position(towers, z)

    # load the data
    data = np.load('./data_motion.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z']

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)

    estimate_motion(towers, z)
