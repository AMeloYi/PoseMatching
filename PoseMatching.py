from skimage.morphology import skeletonize
from skimage import data
import sknw
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import math


def lireFichier(filename):
    start = time.time()
    # open and skeletonize
    img = cv2.imread(filename)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    img1 = cv2.blur(thresh, (5,5))
    for i in range(img1.shape[1]):
        for j in range(img1.shape[0]):
            if img1[j][i] != 255:
                img1[j][i] = 1
                img[j][i] = 0
            else:
                img1[j][i] = 0
                img[j][i] = 255

    # print('open : ', time.time()-start)
    # cv2.imshow('img', img)
    return img1

def skeleton(img1):
    start = time.time()
    ske = skeletonize(img1).astype(np.uint16)
    # build graph from skeleton
    graph = sknw.build_sknw(ske)
    # print('skeleton : ', time.time()-start)
    # draw image
    plt.imshow(img1, cmap='gray')
    return graph

def firstFilter(graph):
    # first filter delete nodes
    start = time.time()
    for i in range(5):
        node_remove_list = []
        for e in graph.edges(data='weight'):
            if(e[2] < 100):
                if(graph.degree(e[1]) == 1):
                    if(graph.degree(e[0]) >= 3):
                        node_remove_list.append(e[1])
                if(graph.degree(e[0]) == 1):
                    if(graph.degree(e[1]) >= 3):
                        node_remove_list.append(e[0])
        for i in range(len(node_remove_list)):
            graph.remove_node(node_remove_list[i])
    # print('first fliter : ', time.time()-start)

def secondFilter(graph):
    start = time.time()
    # second filter delete nodes on arc
    length = len(graph.nodes())
    tmpLength = length
    STOP = False
    while(not STOP):
        node_to_delete = None
        for e in graph.edges(data=True):
            if(e[2]['weight'] < 200):
                if(graph.degree(e[0]) == 2):
                    node_to_delete = e[0]
                    break
                elif(graph.degree(e[1]) == 2):
                    node_to_delete = e[1]
                    break
        edge1, edge2 = None, None
        if(node_to_delete != None):
            for e in graph.edges(data=True):
                if((e[0] == node_to_delete) | (e[1] == node_to_delete)):
                    if (edge1 == None):
                        edge1 = e
                    else:
                        edge2 = e
            if(edge1 != None) & (edge2 != None):
                newWeight = edge1[2]['weight'] + edge2[2]['weight']
                newPath = np.concatenate([edge1[2]['pts'],edge2[2]['pts']])
                strat, end = None, None
                if edge1[0] == node_to_delete:
                    start = edge1[1]
                else:
                    start = edge1[0]
                if edge2[0] == node_to_delete:
                    end = edge2[1]
                else:
                    end = edge2[0]
                graph.remove_node(node_to_delete)
                graph.add_edge(start, end, weight=newWeight, pts=newPath)
        else:
            STOP = True

        if (tmpLength == len(graph.nodes())):
            STOP = True
        else:
            tmpLength = len(graph.nodes())
    # print('second filter : ', time.time()-start)

def thirdFilter(graph):
    start = time.time()
    # third filter
    node_to_delete = None
    node_to_reserve = None
    for e in graph.edges(data=True):
        if(e[2]['weight'] < 105):
            if(graph.degree(e[0]) == 3) & (graph.degree(e[1]) == 3):
                node_to_delete = e[0]
                node_to_reserve = e[1]
    edge1, edge2, edge_reserve = None, None, None
    start1, start2 = None, None
    for e in graph.edges(data=True):
        if node_to_delete in [e[0], e[1]]:
            if node_to_reserve not in [e[0], e[1]]:
                if (edge1 == None):
                    edge1 = e
                    if(node_to_delete == e[0]):
                        start1 = e[1]
                    else:
                        start1 = e[0]
                else:
                    edge2 = e
                    if(node_to_delete == e[0]):
                        start2 = e[1]
                    else:
                        start2 = e[0]
            else:
                edge_reserve = e
    if(node_to_delete in graph.nodes()):
        graph.remove_node(node_to_delete)

        newWeight = edge1[2]['weight'] + edge_reserve[2]['weight']
        newPath = np.concatenate([edge1[2]['pts'],edge_reserve[2]['pts']])
        graph.add_edge(start1, node_to_reserve, weight=newWeight, pts=newPath)

        newWeight = edge2[2]['weight'] + edge_reserve[2]['weight']
        newPath = np.concatenate([edge2[2]['pts'],edge_reserve[2]['pts']])
        graph.add_edge(start2, node_to_reserve, weight=newWeight, pts=newPath)

    # print('third filter : ', time.time()-start)

def deleteCircle(graph):
    edge_to_delete = []
    for e in graph.edges(data=True):
        if(e[2]['weight'] == 0):
            edge_to_delete.append(e)
    for e in edge_to_delete:
        graph.remove_edge(e[0], e[1])

def searchTrunk(graph):
    start = time.time()
    # calculate tezheng
    center1 = None
    center2 = None
    trunk = []
    for n in graph.nodes():
        if(graph.degree(n) == 4):
            center1 = n
        elif(graph.degree(n) == 3):
            center2 = n
        elif(graph.degree(n) == 1):
            trunk.append(n)

    node = graph.node
    if center1 != None:
        posi_center1 = node[center1]['o']
    else:
        print('Cannot find center1')
    if center2 != None:
        posi_center2 = node[center2]['o']
    else:
        print('Cannot find center2')

    # find head, hands and legs
    node_list = []
    distance_list = []
    for n in trunk:
        y = node[n]['o'][0]
        x = node[n]['o'][1]
        if y > posi_center2[0]:
            if x < posi_center2[1]:
                rightleg = n
            if x > posi_center2[1]:
                leftleg = n
        else:
            for e in graph.edges(data=True):
                if n in [e[0], e[1]]:
                    node_list.append(n)
                    distance_list.append(e[2]['weight'])

    head = node_list[distance_list.index(min(distance_list))]

    for n in node_list:
        if n != head:
            x = node[n]['o'][1]
            if x < node[center1]['o'][1]:
                righthand = n
            else:
                lefthand = n
    # print('find hands :', time.time()-start)

    # print('center1 : ', center1)
    # print('center2 : ', center2)
    # print('head : ', head)
    # print('lefthand : ', lefthand)
    # print('righthand : ', righthand)
    # print('leftleg : ', leftleg)
    # print('rightleg : ', rightleg)
    return center1, center2, head, lefthand, righthand, leftleg, rightleg

def drawGraph(graph):
    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')

    # draw node by o
    node, nodes = graph.node, graph.nodes()
    ps = np.array([node[i]['o'] for i in nodes])
    plt.plot(ps[:,1], ps[:,0], 'r.')

    # title and show
    plt.title('Build Graph')
    plt.show()

def calculateWeight(graph, head, lefthand, righthand, leftleg, rightleg):
    for e in graph.edges(data=True):
        if head in [e[0], e[1]]:
            weight_head = e[2]['weight']
        elif lefthand in [e[0], e[1]]:
            weight_lefthand = e[2]['weight']
        elif righthand in [e[0], e[1]]:
            weight_righthand = e[2]['weight']
        elif leftleg in [e[0], e[1]]:
            weight_leftleg = e[2]['weight']
        elif rightleg in [e[0], e[1]]:
            weight_rightleg = e[2]['weight']
    return weight_head, weight_lefthand, weight_righthand, weight_leftleg, weight_rightleg

def calEuclideanDistance(posi1_x, posi1_y, posi2_x, posi2_y):
    a = np.square(posi1_x - posi2_x)
    b = np.square(posi1_y - posi2_y)
    c = a + b
    dist = np.sqrt(c)
    # dist = math.sqrt(sum(np.square(posi1_x - posi2_x), np.square(posi1_y - posi2_y)))
    return dist

def calculateDistance(graph, center1, center2, head, lefthand, righthand, leftleg, rightleg):
    node = graph.node
    center1_x, center1_y = node[center1]['o'][1], node[center1]['o'][0]
    center2_x, center2_y = node[center2]['o'][1], node[center2]['o'][0]
    head_x, head_y = node[head]['o'][1], node[head]['o'][0]
    lefthand_x, lefthand_y = node[lefthand]['o'][1], node[lefthand]['o'][0]
    righthand_x, righthand_y = node[righthand]['o'][1], node[righthand]['o'][0]
    leftleg_x, leftleg_y = node[leftleg]['o'][1], node[leftleg]['o'][0]
    rightleg_x, rightleg_y = node[rightleg]['o'][1], node[rightleg]['o'][0]

    dis_head = calEuclideanDistance(head_x, head_y, center1_x, center1_y)
    dis_lefthand = calEuclideanDistance(lefthand_x, lefthand_y, center1_x, center1_y)
    dis_righthand = calEuclideanDistance(righthand_x, righthand_y, center1_x, center1_y)
    dis_leftleg = calEuclideanDistance(leftleg_x, leftleg_y, center2_x, center2_y)
    dis_rightleg = calEuclideanDistance(rightleg_x, rightleg_y, center2_x, center2_y)
    return dis_head, dis_lefthand, dis_righthand, dis_leftleg, dis_rightleg

def calculateAngle(posi1_x, posi1_y, posi2_x, posi2_y):
    dis = calEuclideanDistance(posi1_x, posi1_y, posi2_x, posi2_y)
    dis_x = abs(posi1_x - posi2_x)
    cos_angle = dis_x / dis
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2

def calculateAngles(graph, center1, center2, head, lefthand, righthand, leftleg, rightleg):
    node = graph.node
    center1_x, center1_y = node[center1]['o'][1], node[center1]['o'][0]
    center2_x, center2_y = node[center2]['o'][1], node[center2]['o'][0]
    head_x, head_y = node[head]['o'][1], node[head]['o'][0]
    lefthand_x, lefthand_y = node[lefthand]['o'][1], node[lefthand]['o'][0]
    righthand_x, righthand_y = node[righthand]['o'][1], node[righthand]['o'][0]
    leftleg_x, leftleg_y = node[leftleg]['o'][1], node[leftleg]['o'][0]
    rightleg_x, rightleg_y = node[rightleg]['o'][1], node[rightleg]['o'][0]

    angle_head = calculateAngle(head_x, head_y, center1_x, center1_y)
    angle_lefthand = calculateAngle(lefthand_x, lefthand_y, center1_x, center1_y)
    angle_righthand = calculateAngle(righthand_x, righthand_y, center1_x, center1_y)
    angle_leftleg = calculateAngle(leftleg_x, leftleg_y, center2_x, center2_y)
    angle_rightleg = calculateAngle(rightleg_x, rightleg_y, center2_x, center2_y)
    return angle_head, angle_lefthand, angle_righthand, angle_leftleg, angle_rightleg

def calculateDataset(graph):
    center1, center2, head, lefthand, righthand, leftleg, rightleg = searchTrunk(graph)
    node = graph.node
    # data = [X, Y, weight, distance, degree]
    data_head = [node[head]['o'][1], node[head]['o'][0]]
    data_lefthand = [node[lefthand]['o'][1], node[lefthand]['o'][0]]
    data_righthand = [node[righthand]['o'][1], node[righthand]['o'][0]]
    data_leftleg = [node[leftleg]['o'][1], node[leftleg]['o'][0]]
    data_rightleg = [node[rightleg]['o'][1], node[rightleg]['o'][0]]

    weight_head, weight_lefthand, weight_righthand, weight_leftleg, weight_rightleg = calculateWeight(graph, head, lefthand, righthand, leftleg, rightleg)
    data_head.append(weight_head)
    data_lefthand.append(weight_lefthand)
    data_righthand.append(weight_righthand)
    data_leftleg.append(weight_leftleg)
    data_rightleg.append(weight_rightleg)

    dis_head, dis_lefthand, dis_righthand, dis_leftleg, dis_rightleg = calculateDistance(graph, center1, center2, head, lefthand, righthand, leftleg, rightleg)
    data_head.append(dis_head)
    data_lefthand.append(dis_lefthand)
    data_righthand.append(dis_righthand)
    data_leftleg.append(dis_leftleg)
    data_rightleg.append(dis_rightleg)

    angle_head, angle_lefthand, angle_righthand, angle_leftleg, angle_rightleg = calculateAngles(graph, center1, center2, head, lefthand, righthand, leftleg, rightleg)
    data_head.append(angle_head)
    data_lefthand.append(angle_lefthand)
    data_righthand.append(angle_righthand)
    data_leftleg.append(angle_leftleg)
    data_rightleg.append(angle_rightleg)

    return data_head, data_lefthand, data_righthand, data_leftleg, data_rightleg

def learning():
    files = []
    for i in [1,3,5]:
        for j in range(1,8):
            filename = r'.\image\bdd\pose' + str(j) + '_' + str(i) + '.png'
            files.append([filename, j])
    data = []
    for file in files:
        print(file[0])
        nbClass = file[1]
        img = lireFichier(file[0])
        graph = skeleton(img)
        firstFilter(graph)
        secondFilter(graph)
        thirdFilter(graph)
        deleteCircle(graph)
        data_head, data_lefthand, data_righthand, data_leftleg, data_rightleg = calculateDataset(graph)
        percentage_length_left = data_lefthand[3] / data_lefthand[2]
        percentage_angle_left = data_lefthand[4] / 90
        percentage_length_right = data_righthand[3] / data_righthand[2]
        percentage_angle_right = data_righthand[4] / 90
        data.append([percentage_length_left, percentage_angle_left, percentage_length_right, percentage_angle_right, nbClass])
        # data.append([data_head, data_lefthand, data_righthand, data_leftleg, data_rightleg])
        # drawGraph(graph)
    return data

def classify(dataset, file, k):
    img = lireFichier(file)
    graph = skeleton(img)
    firstFilter(graph)
    secondFilter(graph)
    thirdFilter(graph)
    deleteCircle(graph)
    data_head, data_lefthand, data_righthand, data_leftleg, data_rightleg = calculateDataset(graph)
    percentage_length_left = data_lefthand[3] / data_lefthand[2]
    percentage_angle_left = data_lefthand[4] / 90
    percentage_length_right = data_righthand[3] / data_righthand[2]
    percentage_angle_right = data_righthand[4] / 90
    data = [percentage_length_left, percentage_angle_left, percentage_length_right, percentage_angle_right]
    #calculate all the distances
    disList = []
    for v in dataset:
        nbClass = v[4]
        d = np.linalg.norm(np.array(v[:4]) - np.array(data))
        disList.append([nbClass, d])
    #sort
    disList.sort(key = lambda dis: dis[1])
    #choose k min dataset
    disList=disList[:k]
    countClass = [0, 0, 0, 0, 0, 0, 0]
    for e in disList:
        countClass[e[0]-1] += 1
    maxClass = countClass.index(max(countClass)) + 1
    return maxClass




if __name__ == '__main__':
    # dataset = learning()
    # print(dataset)
    dataset = [[0.95538849926501679, 0.12850449403831016, 0.95607555761373386, 0.10993597172811036, 1],
    [0.70109683150419499, 0.5997624990749979, 0.92500628194186574, 0.034936152581266394, 2],
    [0.93451954806905446, 0.091877544275427173, 0.71953498221662593, 0.54638285415302024, 3],
    [0.7425215208452115, 0.55693255171171985, 0.68135745141528647, 0.51261856296643815, 4],
    [0.96147564687258169, 0.43020218568948804, 0.93866505867482064, 0.060189134598406281, 5],
    [0.93211365928796241, 0.056000150237612088, 0.96906465887659754, 0.47413538789954274, 6],
    [0.96579130053242246, 0.43908430797129283, 0.96916678579564197, 0.4456448932536004, 7],
    [0.92738648115915789, 0.054679833270935514, 0.91718759959604734, 0.024473312908508908, 1],
    [0.79826416731085481, 0.5839764320905958, 0.9369419871564062, 0.10316852658287969, 2],
    [0.91582128046321376, 0.10662465609255049, 0.69165965748024494, 0.64704903973486971, 3],
    [0.81132737591138748, 0.50224424371983134, 0.78981529372108916, 0.58830930225896427, 4],
    [0.96050926841198947, 0.57240626023638674, 0.89152987050837096, 0.019522092332315454, 5],
    [0.91275687047452714, 0.029351127963428203, 0.93709835810344533, 0.52522721856294552, 6],
    [0.97260897072241959, 0.47629189851506598, 0.95080160870360486, 0.53096007404806467, 7],
    [0.95258314701854307, 0.10491536487659832, 0.94525181442595485, 0.1204105065470735, 1],
    [0.72076033869868505, 0.60144640535865201, 0.94042164731186129, 0.07734403362498371, 2],
    [0.94130078231067005, 0.057715876752608884, 0.7486057938155618, 0.58737805328332826, 3],
    [0.79108678647362785, 0.46518934406618534, 0.75428863539069035, 0.49040946724061912, 4],
    [0.96849479332861399, 0.45074563165912529, 0.94012214939560224, 0.098108274087878491, 5],
    [0.92330387057178165, 0.06615674923516697, 0.95760391858329741, 0.47180601021114554, 6],
    [0.97547663718750333, 0.44042701132631396, 0.94553099990460598, 0.53254660014344912, 7]]

    mxConfusion = np.zeros(49)
    mxConfusion = mxConfusion.reshape(7, 7)

    files = []
    for i in [2,4,6]:
        for j in range(1,8):
            filename = r'.\image\bdd\pose' + str(j) + '_' + str(i) + '.png'
            files.append([filename, j])
    for file in files:
        nbClass = classify(dataset, file[0], 3)
        print(file[0], ' is classified to ', nbClass)
        mxConfusion[file[1] - 1][nbClass - 1] += 1

    print(mxConfusion)
