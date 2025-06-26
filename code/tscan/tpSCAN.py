# -*- coding: utf-8 -*-
import readfile
import math
import sys
import os
# import matplotlib.pyplot as plt
import networkx as nx
from time import sleep, time
import datetime
import json
import gc
import itertools
import pprint
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

class tGraph:
    def __init__(self, path):
        self.path = path
        self.rank = []
        self.toprank = 0
        self.visited_set = []
        self.visited_node = []
        self.visited_edge = []
        self.eps = 0.3
        self.miu = 5
        self.tau = 3
        self.theta = 1
        self.sigma = {}
        self.sigma_t = {}
        self.union_set = []
        self.collections = {}
        self.adj = {}
        self.subgraph = {}
        self.frquent_set = {}
        self.G = readfile.tGraph(self.path)

        # print("#Nodes: " + str(len(self.G.nodes())))
        # print("#Edges: " + str(len(self.G.edges())))
        temporal_edge = 0
        for nodex in self.G.nodes():
            for nodey in self.G.adj[nodex]:
                temporal_edge += len(self.G.edges[nodex, nodey]['t'])
        # print("Temporal edges:" + str(temporal_edge / 2))

        ranktemp = {}
        for node_temp in self.G.nodes():
            self.G.nodes[node_temp]['l'] = 0
            self.G.nodes[node_temp]['u'] = len(self.G.adj[node_temp])
            ranktemp[node_temp] = self.G.nodes[node_temp]['u']
        self.rank = sorted(ranktemp.items(), key=lambda item: item[1], reverse=True)

        for node_temp in ranktemp:
            self.adj[node_temp] = {}
            for item in self.G.adj[node_temp]:
                if item == node_temp:
                    print(item)
                else:
                    self.adj[node_temp][item] = ranktemp[item]

            adjtemp = sorted(self.adj[node_temp].items(), key=lambda item: item[1], reverse=True)

            self.adj[node_temp] = []
            for item in adjtemp:
                self.adj[node_temp].append(item[0])

        ranktemp = []
        self.toprank = self.rank[0]
        for i in self.rank:
            ranktemp.append(i[0])
        self.rank = ranktemp

        del(ranktemp)

    def tDistribution(self, tempG):
        timestamps = {}
        for item in tempG.edges.data():
            if item[2]['t'] in timestamps:
                timestamps[item[2]['t']] = timestamps[item[2]['t']] + 1
            else:
                timestamps[item[2]['t']] = 1

        min = -1
        max = -1
        x = []
        y = []
        for k, v in timestamps.items():
            if k > max:
                max = k
            if min > k or min == -1:
                min = k
            x.append(k)
            y.append(v)
        # print (min, max)
        #print(time.gmtime(min), time.gmtime(max))

    def check_SCANB_core(self, u):
        if self.G.nodes[u]['l'] >= self.miu:
            return True

        if self.G.nodes[u]['u'] < self.miu:
            return False

        if self.G.nodes[u]['l'] < self.miu and self.G.nodes[u]['u'] >= self.miu:
            if self.frquent_mining(u):
                self.G.nodes[u]['l'] = self.miu
                return True
            else:
                self.G.nodes[u]['u'] = 0
                return False

    def cluster_SCANB_core(self, u):
        candidate_set = []
        for v in self.G.adj[u]:
            if len(self.G.edges[u, v]['t']) >= self.tau:
                candidate_set.append(v)

        for v in candidate_set:
            if v >= u:
                edge_set = (u, v)
            else:
                edge_set = (v, u)

            if edge_set in self.sigma:
                if self.G.nodes[v]['l'] >= self.miu and self.sigma[edge_set] >= self.tau:
                    self.union(u, v)
            else:
                if self.G.nodes[v]['u'] >= self.miu:
                    self.sigma[edge_set] = self.compute_sigma(u, v)
                    self.check_SCANB_core(v)
                    if self.G.nodes[v]['l'] >= self.miu and self.sigma[edge_set] >= self.tau:
                        self.union(u, v)

    def check_SCANW_core(self, u):
        if self.G.nodes[u]['l'] < self.miu and self.G.nodes[u]['u'] >= self.miu:
            for v in self.G.adj[u]:
                if v >= u:
                    edge_set = (u, v)
                else:
                    edge_set = (v, u)

                if edge_set not in self.sigma:
                    self.sigma[edge_set] = self.compute_sigma(u, v)

                    if self.sigma[edge_set] >= self.tau:
                        self.G.nodes[u]['l'] += 1
                        self.G.nodes[v]['l'] += 1
                    else:
                        self.G.nodes[u]['u'] -= 1
                        self.G.nodes[v]['u'] -= 1

                    if self.G.nodes[u]['l'] >= self.miu or self.G.nodes[u]['u'] < self.miu:
                        break

        self.visited_node.append(u)

    def cluster_SCANW_core(self, u):
        candidate_set = []
        for v in self.G.adj[u]:
            if len(self.G.edges[u, v]['t']) >= self.tau:
                candidate_set.append(v)

        for v in candidate_set:
            if v >= u:
                edge_set = (u, v)
            else:
                edge_set = (v, u)

            if edge_set in self.sigma:
                if self.G.nodes[v]['l'] >= self.miu and self.sigma[edge_set] >= self.tau:
                    self.union(u, v)
            else:
                if self.G.nodes[v]['u'] >= self.miu:  # and
                    self.sigma[edge_set] = self.compute_sigma(u, v)

                    if self.sigma[edge_set] >= self.tau:
                        self.G.nodes[u]['l'] += 1
                        self.G.nodes[v]['l'] += 1
                    else:
                        self.G.nodes[u]['u'] -= 1
                        self.G.nodes[v]['u'] -= 1

                    if self.G.nodes[v]['l'] >= self.miu and self.sigma[edge_set] >= self.tau:
                        self.union(u, v)

    def check_SCANS_core(self, u):
        if self.G.nodes[u]['l'] < self.miu and self.G.nodes[u]['u'] >= self.miu:
            candidate_set = []
            for v in self.G.adj[u]:
                if len(self.G.edges[u, v]['t']) >= self.tau:
                    candidate_set.append(v)
            if len(candidate_set) < self.miu:
                return False

            candidate_time = {}
            for v in candidate_set:
                for time_item in self.G.edges[u, v]['t']:
                    if time_item in candidate_time:
                        candidate_time[time_item] += 1
                    else:
                        candidate_time[time_item] = 1

            times_more_than_miu = []
            for key in candidate_time:
                if candidate_time[key] >= self.miu:
                    times_more_than_miu.append(key)
            if len(times_more_than_miu) < self.tau:
                return False

            tau_calculate = 0
            for t in times_more_than_miu:
                miu_calculate = 0
                max_miu_calculate = candidate_time[t]
                for v in candidate_set:
                    if tau_calculate >= self.tau:
                        break
                    if v >= u:
                        edge_set = (u, v, t)
                    else:
                        edge_set = (v, u, t)
                    if edge_set not in self.sigma_t:
                        self.sigma_t[edge_set] = self.compute_sigma_at_one_time(u, v, t)

                    if self.sigma_t[edge_set] >= self.eps:
                        miu_calculate += 1
                    else:
                        max_miu_calculate -= 1

                    if miu_calculate >= self.miu:
                        tau_calculate += 1
                        continue

                    if max_miu_calculate < self.miu:
                        continue

            if tau_calculate < self.tau:
                self.G.nodes[u]['l'] = 0
                self.visited_node.append(u)
                return False

            for v in candidate_set:
                if v >= u:
                    edge_set = (u, v)
                else:
                    edge_set = (v, u)

                if edge_set not in self.sigma:
                    self.sigma[edge_set] = self.compute_sigma(u, v)

                    if self.sigma[edge_set] >= self.tau:
                        self.G.nodes[u]['l'] += 1
                        self.G.nodes[v]['l'] += 1
                    else:
                        self.G.nodes[u]['u'] -= 1
                        self.G.nodes[v]['u'] -= 1

                    if self.G.nodes[u]['l'] >= self.miu or self.G.nodes[u]['u'] < self.miu:
                        if self.frquent_mining(u):
                            self.G.nodes[u]['l'] = self.miu
                        else:
                            self.G.nodes[u]['l'] = 0
                        break

        self.visited_node.append(u)
        return False

    def cluster_SCANS_core(self, u):
        candidate_set = []
        for v in self.G.adj[u]:
            if len(self.G.edges[u, v]['t']) >= self.tau:
                candidate_set.append(v)

        for v in candidate_set:
            if v >= u:
                edge_set = (u, v)
            else:
                edge_set = (v, u)

            if edge_set in self.sigma:
                if self.G.nodes[v]['l'] >= self.miu and self.sigma[edge_set] >= self.tau:
                    self.union(u, v)
            else:
                if self.G.nodes[v]['u'] >= self.miu:
                    self.sigma[edge_set] = self.compute_sigma(u, v)
                    if self.sigma[edge_set] >= self.tau:
                        self.G.nodes[u]['l'] += 1
                        self.G.nodes[v]['l'] += 1
                    else:
                        self.G.nodes[u]['u'] -= 1
                        self.G.nodes[v]['u'] -= 1

                    if self.sigma[edge_set] >= self.tau:
                        self.check_SCANS_core(v)
                        if self.G.nodes[v]['l'] >= self.miu:
                            self.union(u, v)
                            self.visited_node.append(u)

    def check_SCANA_core(self, u):
        if self.G.nodes[u]['l'] < self.miu and self.G.nodes[u]['u'] >= self.miu:
            candidate_set = []
            for v in self.G.adj[u]:
                if len(self.G.edges[u, v]['t']) >= self.tau:
                    candidate_set.append(v)
            if len(candidate_set) < self.miu:
                return False

            candidate_time = {}
            for v in candidate_set:
                for time_item in self.G.edges[u, v]['t']:
                    if time_item in candidate_time:
                        candidate_time[time_item] += 1
                    else:
                        candidate_time[time_item] = 1

            times_more_than_miu = []
            for key in candidate_time:
                if candidate_time[key] >= self.miu:
                    times_more_than_miu.append(key)
            if len(times_more_than_miu) < self.tau:
                return False

            tau_calculate = 0
            for t in times_more_than_miu:
                miu_calculate = 0
                max_miu_calculate = candidate_time[t]
                for v in candidate_set:
                    if tau_calculate >= self.tau:
                        break
                    if v >= u:
                        edge_set = (u, v, t)
                    else:
                        edge_set = (v, u, t)
                    if edge_set not in self.sigma_t:
                        self.sigma_t[edge_set] = self.compute_sigma_at_one_time(u, v, t)

                    if self.sigma_t[edge_set] >= self.eps:
                        miu_calculate += 1
                    else:
                        max_miu_calculate -= 1

                    if miu_calculate >= self.miu:
                        tau_calculate += 1
                        continue

                    if max_miu_calculate < self.miu:
                        continue

            if tau_calculate < self.tau:
                self.G.nodes[u]['l'] = 0
                self.visited_node.append(u)
                return False

            for v in candidate_set:
                if v >= u:
                    edge_set = (u, v)
                else:
                    edge_set = (v, u)

                if edge_set not in self.sigma:
                    self.sigma[edge_set] = self.compute_sigma(u, v)

                    if self.sigma[edge_set] >= self.tau:
                        self.G.nodes[u]['l'] += 1
                        self.G.nodes[v]['l'] += 1
                    else:
                        self.G.nodes[u]['u'] -= 1
                        self.G.nodes[v]['u'] -= 1

                    if self.G.nodes[u]['l'] >= self.miu or self.G.nodes[u]['u'] < self.miu:
                        break

        self.visited_node.append(u)
        return False

    def cluster_SCANA_core(self, u):
        candidate_set = []
        for v in self.G.adj[u]:
            if len(self.G.edges[u, v]['t']) >= self.tau:
                candidate_set.append(v)

        for v in candidate_set:
            if v >= u:
                edge_set = (u, v)
            else:
                edge_set = (v, u)

            if edge_set in self.sigma:
                if self.G.nodes[v]['l'] >= self.miu and self.sigma[edge_set] >= self.tau:
                    self.union(u, v)
            else:
                if self.G.nodes[v]['u'] >= self.miu:
                    self.sigma[edge_set] = self.compute_sigma(u, v)
                    if self.sigma[edge_set] >= self.tau:
                        self.G.nodes[u]['l'] += 1
                        self.G.nodes[v]['l'] += 1
                    else:
                        self.G.nodes[u]['u'] -= 1
                        self.G.nodes[v]['u'] -= 1

                    if self.sigma[edge_set] >= self.tau:
                        self.check_SCANA_core(v)
                        if self.G.nodes[v]['l'] >= self.miu:
                            self.union(u, v)
                            self.visited_node.append(u)

    def add_node_set(self, u):
        if len(self.union_set):
            for set in self.union_set:
                if u in set:
                    return 0
                if u not in set:
                    pass
            self.union_set.append([u])
        else:
            self.union_set.append([u])

    def union(self, u, v):
        if len(self.union_set):
            flag = 0
            set1 = []
            set2 = []
            for set in self.union_set:
                if u in set and v in set:
                    flag = -1  # no need to change
                    break
                if u in set and v not in set:
                    set1 = set
                    flag = flag + 1
                if v in set and u not in set:
                    set2 = set
                    flag = flag + 1
            if flag == 0:
                temp = [u, v]
                self.union_set.append(temp)
            if flag == 1:
                if set1:
                    index_temp = self.union_set.index(set1)
                    self.union_set[index_temp].append(v)
                if set2:
                    index_temp = self.union_set.index(set2)
                    self.union_set[index_temp].append(u)
            if flag == 2:
                self.union_set.remove(set1)
                self.union_set.remove(set2)
                union_temp = set1 + set2
                self.union_set.append(union_temp)
            if flag > 2:
                print("unnion error")

        else:
            temp = [u, v]
            self.union_set.append(temp)

    def compute_sigma_at_one_time(self, u, v, t):
        adju = []
        for vertex in self.adj[u]:
            if t in self.G.edges[u, vertex]['t']:
                adju.append(vertex)

        adjv = []
        for vertex in self.adj[v]:
            if t in self.G.edges[v, vertex]['t']:
                adjv.append(vertex)

        lenuadj = len(adju) + 1
        lenvadj = len(adjv) + 1
        if lenuadj < self.eps * self.eps * lenvadj or lenvadj < self.eps * self.eps * lenuadj:
            # print(u,v,lenuadj,lenuadj)
            return 0

        len_v_u = len(set(adju) & set(adjv)) + 2
        if len_v_u < self.eps * math.sqrt(lenuadj * lenvadj):
            # print(u, v, len_v_u, lenuadj, lenuadj)
            return 0
        else:
            return self.eps + 0.1

    def compute_sigma(self, u, v):
        tau = 0

        if len(self.G.edges[u, v]['t']) < self.tau:
            return 0

        for t in self.G.edges[u, v]['t']:
            if v >= u:
                edge_set = (u, v, t)
            else:
                edge_set = (v, u, t)

            if edge_set not in self.sigma_t:
                result = self.compute_sigma_at_one_time(u, v, t)
                # print u,v,t,result
                self.sigma_t[edge_set] = result
                if result > self.eps:
                    tau += 1
            else:
                if self.sigma_t[edge_set] > self.eps:
                    tau += 1

            if tau >= self.tau:
                return tau
        return 0

    def compute_sigma2(self, u, v):
        tau = 0
        if len(self.G.edges[u, v]['t']) < self.tau:
            return 0

        for t in self.G.edges[u, v]['t']:
            if t in self.subgraph:
                pass
            else:
                self.subgraph[t] = subgraph.subgraph(self.path, t, self.theta)
                #print(t)
            if self.subgraph[t].compute_sigma(u, v, self.eps) > self.eps:
                tau = tau + 1
            if tau >= self.tau:
                return tau
        return 0

    def print_degree(self, top_percents_list):
        length = 0
        length2 = 0
        for item in self.G.edges():
            length += len(self.G.edges[item[0], item[1]]['t'])
        #print("temporal edges:" + str(length))

        for item in self.G[self.toprank[0]]:
            length2 += len(self.G.edges[item, self.toprank[0]]['t'])

        #print(self.toprank)
        #print("max degree (including temporal edges):" + str(length2))

        for top_percents in top_percents_list:
            flag = int(len(self.G.edges) / 100 * top_percents)
            degree = 0
            degree2 = 0
            # print(flag)
            for item in self.rank:
                if flag <= 0:
                    break
                flag -= 1
                lengtemp = 0
                for item2 in self.G[item]:
                    lengtemp += len(self.G.edges[item, item2]['t'])
                degree += lengtemp
                degree2 += len(self.G[item])
            #print("top%:" + str(top_percents))
            #print("temporal degree:" + str(degree) + "####" + str(degree / int(len(self.G.edges) / 100 * top_percents)))
            #print("static degree:" + str(degree2) + "####" + str(degree2 / int(len(self.G.edges) / 100 * top_percents)))

    def separability(self, datapath, clusterpath):
        temp = []
        for line in open(clusterpath):
            temp.append(json.loads(line))

        as_all = 0.0

        for setx in temp:
            # print setx
            flag = 0.0
            flag2 = 0
            for v in setx:
                for u in set(self.adj[v]) & set(setx):
                    flag += len(self.G.edges[u, v]['t'])
                for u in self.adj[v]:
                    flag2 += len(self.G.edges[u, v]['t'])
                if (flag2 - flag) == 0:
                    as_single = 1
                else:
                    as_single = flag / (flag2 - flag)

            as_all += as_single

        #print(as_all, len(temp), as_all / len(temp))
        try:
            return as_all / len(temp)
        except ZeroDivisionError:
            return None
        
    def density(self, datapath, clusterpath):
        temp = []
        for line in open(clusterpath):
            temp.append(json.loads(line))

        ad_all = 0.0

        for set in temp:
            flag = 0.0
            flag2 = 0
            for item in set:
                for v in self.adj[item]:
                    flag += len(self.G.edges[item, v]['t'])
                ad_single = flag / len(set)
            ad_all += ad_single
        try:    
            ad_avg = ad_all / len(temp)
        except ZeroDivisionError:
            ad_avg = None
        return ad_avg

    def cohesiveness(self, datapath, clusterpath):
        temp = []
        for line in open(clusterpath):
            temp.append(json.loads(line))

        max = 0.0
        for setx in temp:
            # print setx
            flag = 0.0
            flag2 = 0
            for v in setx:
                for u in set(self.adj[v]) & set(setx):
                    flag += len(self.G.edges[u, v]['t'])
                for u in self.adj[v]:
                    flag2 += len(self.G.edges[u, v]['t'])
                if (flag2 - flag) == 0:
                    as_single = 1
                else:
                    as_single = flag / (flag2 - flag)

            if max < as_single:
                max = as_single

        #print('ac', max)
        return max

    def ccoefficient(self, datapath, clusterpath):
        temp = []
        for line in open(clusterpath):
            temp.append(json.loads(line))

        flag = 0
        flag2 = 0
        flag3 = 0
        result = []
        for set in temp:
            for item in set:
                for v in self.adj[item]:
                    flag += len(self.G.edges[item, v]['t'])
                    flag3 += 1

                listtemp = itertools.combinations(self.adj[item], 2)
                setlists = list(listtemp)

                for v_set1 in setlists:
                    if v_set1 in self.G.edges():
                        flag2 += len(self.G.edges[v_set1]['t'])

                resulttemp = 2.0 * flag2 / flag

                result.append(resulttemp)

        #print(sum(result), len(result), sum(result) / len(result))
        return sum(result) / len(result)

    def separability_by_year(self, datapath, clusterpath):
        temp = []
        for line in open(clusterpath):
            temp.append(json.loads(line))

        for year in range(50, 79):

            as_all = 0.0
            for setx in temp:
                # print setx
                flag = 0.0
                flag2 = 0
                for v in setx:
                    for u in set(self.adj[v]) & set(setx):
                        if year in self.G.edges[u, v]['t']:
                            flag += 1
                    for u in self.adj[v]:
                        if year in self.G.edges[u, v]['t']:
                            flag2 += 1
                    if (flag2 - flag) == 0:
                        as_single = 1
                    else:
                        as_single = flag / (flag2 - flag)
                as_all += as_single
            #print(year, as_all, len(temp), as_all / len(temp))

    def density_by_year(self, datapath, clusterpath):
        temp = []
        for line in open(clusterpath):
            temp.append(json.loads(line))

        for year in range(50, 79):
            flag = 0.0
            flag2 = 0
            for set in temp:
                for item in set:
                    for v in self.adj[item]:
                        if year in self.G.edges[item, v]['t']:
                            flag += 1
                    flag2 += 1
            #print(year, flag, flag2, flag / flag2)

    def cohesiveness_by_year(self, datapath, clusterpath):
        temp = []
        for line in open(clusterpath):
            temp.append(json.loads(line))

        for year in range(50, 79):
            max = 0.0
            for setx in temp:
                # print setx
                flag = 0.0
                flag2 = 0
                for v in setx:
                    for u in set(self.adj[v]) & set(setx):
                        if year in self.G.edges[u, v]['t']:
                            flag += 1
                    for u in self.adj[v]:
                        if year in self.G.edges[u, v]['t']:
                            flag2 += 1
                    if (flag2 - flag) == 0:
                        as_single = 1
                    else:
                        as_single = flag / (flag2 - flag)
                if max < as_single:
                    max = as_single

            #print(year, max)

    def ccoefficient_by_year(self, datapath, clusterpath):
        temp = []
        for line in open(clusterpath):
            temp.append(json.loads(line))

        for year in range(50, 79):
            flag = 0
            flag2 = 0
            flag3 = 0
            result = []
            for set in temp:
                for item in set:
                    for v in self.adj[item]:
                        # flag += len(self.G.edges[item,v]['t'])
                        if year in self.G.edges[item, v]['t']:
                            flag3 += 1

                    if flag3 > 0:
                        listtemp = itertools.combinations(self.adj[item], 2)
                        setlists = list(listtemp)

                        for v_set1 in setlists:
                            if v_set1 in self.G.edges():
                                if year in self.G.edges[v_set1]['t']:
                                    # flag2 += len(self.G.edges[v_set1]['t'])
                                    flag2 += 1

                        resulttemp = 2.0 * flag2 / flag3
                        result.append(resulttemp)
                    else:
                        result.append(0)

            #print(year, sum(result), len(result), sum(result) / len(result))

    def modularity(self, datapath, clusterpath):
        for line in open(clusterpath):
            temp = json.loads(line)

        flag = 0.0
        flag2 = 0
        for setx in temp:
            for item in setx:
                hubs = []
                for v in self.adj[item]:
                    if self.compute_sigma(item, v) >= self.tau:
                        hubs.append(v)

            cluster = set(hubs + setx)
            listtemp = itertools.combinations(cluster, 2)
            setlists = list(listtemp)
            for v_set1 in setlists:
                if v_set1 in self.G.edges():
                    flag2 += len(self.G.edges[v_set1]['t'])

            for v in cluster:
                for adj in self.adj[v]:
                    flag += len(self.G.edges[adj, v]['t'])
                flag2 += 1

    def SCANB(self, miu, tau, eps):
        self.eps = eps
        self.miu = miu
        self.tau = tau
        self.union_set = []

        for u in self.G.nodes():
            self.G.nodes[u]['l'] = 0
            self.G.nodes[u]['u'] = len(self.G.adj[u])

        starttime = datetime.datetime.now()
        for u in self.rank:
            value = self.check_SCANB_core(u)
            if value:
                self.add_node_set(u)
                self.cluster_SCANB_core(u)

        endtime = datetime.datetime.now()
        interval = (endtime - starttime).total_seconds()
        #print("Runing time of SCANB:" + str(interval))
        self.write_runtime(interval, sys._getframe().f_code.co_name)
        self.sigma = {}
        self.sigma_t = {}
        self.visited_node = []

        file_name = self.path  + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANB'
        #print("Cores output at: " + file_name)
        file_object = open(file_name, 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()
        self.union_set = []

    def SCANB_slow(self, miu, tau, eps):
        self.eps = eps
        self.miu = miu
        self.tau = tau
        self.union_set = []

        for u in self.G.nodes():
            self.G.nodes[u]['l'] = 0
            self.G.nodes[u]['u'] = len(self.G.adj[u])

        starttime = datetime.datetime.now()
        for u in self.rank:
            value = self.check_SCANB_core(u)
            if value:
                self.union_set.append(u)

        endtime = datetime.datetime.now()
        interval = (endtime - starttime).total_seconds()
        #print("SCANB_slow:" + str(interval))
        self.write_runtime(interval, sys._getframe().f_code.co_name)
        self.sigma = {}
        self.sigma_t = {}
        self.visited_node = []

        file_name = self.path + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANB-'
        #print("Cores output at: " + file_name)
        file_object = open(file_name, 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()
        self.union_set = []

    def SCANW(self, miu, tau, eps):
        self.eps = eps
        self.miu = miu
        self.tau = tau
        self.union_set = []

        starttime = datetime.datetime.now()
        for u in self.G.nodes():
            self.G.nodes[u]['l'] = 0
            self.G.nodes[u]['u'] = len(self.G.adj[u])

        for u in self.rank:
            if self.G.nodes[u]['l'] >= self.miu:
                self.union_set.append(u)

            if self.G.nodes[u]['l'] < miu and self.G.nodes[u]['u'] >= miu:
                for v in self.G.adj[u]:
                    if v >= u:
                        edge_set = (u, v)
                    else:
                        edge_set = (v, u)
                    if edge_set not in self.sigma:
                        self.sigma[edge_set] = self.compute_sigma(u, v)
                        if self.sigma[edge_set] >= self.tau:
                            self.G.nodes[u]['l'] += 1
                            self.G.nodes[v]['l'] += 1
                        else:
                            self.G.nodes[u]['u'] -= 1
                            self.G.nodes[v]['u'] -= 1
                    if self.G.nodes[u]['l'] >= self.miu:
                        self.union_set.append(u)
                    if self.G.nodes[u]['l'] >= self.miu or self.G.nodes[u]['u'] < self.miu:
                        break

        endtime = datetime.datetime.now()
        interval = (endtime - starttime).total_seconds()
        #print("Runing time of SCANW:" + str(interval))
        self.write_runtime(interval, sys._getframe().f_code.co_name)

        self.sigma = {}
        self.sigma_t = {}
        self.visited_node = []

        file_name = self.path + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANW'
        #print("Cores output at: " + file_name)
        file_object = open(file_name, 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()
        self.union_set = []

    def SCANS(self, miu, tau, eps):
        self.eps = eps
        self.miu = miu
        self.tau = tau
        self.union_set = []

        starttime = datetime.datetime.now()
        for u in self.G.nodes():
            self.G.nodes[u]['l'] = 0
            self.G.nodes[u]['u'] = len(self.G.adj[u])

        for u in self.rank:
            if self.G.nodes[u]['l'] >= self.miu:
                self.union_set.append(u)

            if self.G.nodes[u]['l'] < miu and self.G.nodes[u]['u'] >= miu:
                for v in self.G.adj[u]:
                    if v >= u:
                        edge_set = (u, v)
                    else:
                        edge_set = (v, u)
                    if edge_set not in self.sigma:
                        self.sigma[edge_set] = self.compute_sigma(u, v)
                        if self.sigma[edge_set] >= self.tau:
                            self.G.nodes[u]['l'] += 1
                            self.G.nodes[v]['l'] += 1
                        else:
                            self.G.nodes[u]['u'] -= 1
                            self.G.nodes[v]['u'] -= 1
                    if self.G.nodes[u]['l'] >= self.miu:
                        self.union_set.append(u)
                    if self.G.nodes[u]['l'] >= self.miu or self.G.nodes[u]['u'] < self.miu:
                        break

        nodesx = self.union_set[:]
        self.union_set = []

        for u in nodesx:
            candidate_set = []
            for v in set(self.G.adj[u].keys()):  # & set(nodes):
                if len(self.G.edges[u, v]['t']) >= self.tau:
                    candidate_set.append(v)
            if len(candidate_set) < self.miu:
                self.G.nodes[u]['l'] = 0
                continue

            candidate_time = {}
            for v in candidate_set:
                for time_item in self.G.edges[u, v]['t']:
                    if time_item in candidate_time:
                        candidate_time[time_item] += 1
                    else:
                        candidate_time[time_item] = 1

            times_more_than_miu = []
            for key in candidate_time:
                if candidate_time[key] >= self.miu:
                    times_more_than_miu.append(key)
            if len(times_more_than_miu) < self.tau:
                self.G.nodes[u]['l'] = 0
                continue

            tau_calculate = 0
            for t in times_more_than_miu:
                miu_calculate = 0
                max_miu_calculate = candidate_time[t]
                for v in candidate_set:
                    if tau_calculate >= self.tau:
                        break
                    if v >= u:
                        edge_set = (u, v, t)
                    else:
                        edge_set = (v, u, t)
                    if edge_set not in self.sigma_t:
                        self.sigma_t[edge_set] = self.compute_sigma_at_one_time(u, v, t)

                    if self.sigma_t[edge_set] >= self.eps:
                        miu_calculate += 1

                    if miu_calculate >= self.miu:
                        tau_calculate += 1
                        break

                    if max_miu_calculate < self.miu:
                        break

            if tau_calculate < self.tau:
                self.G.nodes[u]['l'] = 0

            if tau_calculate >= self.tau:
                self.G.nodes[u]['l'] = self.tau + 1
                self.union_set.append(u)

        endtime = datetime.datetime.now()
        interval = (endtime - starttime).total_seconds()
        #print("Runing time of SCANS:" + str(interval))
        self.write_runtime(interval, sys._getframe().f_code.co_name)
        self.sigma = {}
        self.sigma_t = {}
        self.visited_node = []
        file_name = self.path + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANS'
        #print("Cores output at: " + file_name)
        file_object = open(file_name, 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()
        self.union_set = []

    def SCANA(self, miu, tau, eps):
        self.eps = eps
        self.miu = miu
        self.tau = tau
        self.union_set = []

        for u in self.G.nodes():
            self.G.nodes[u]['l'] = 0
            self.G.nodes[u]['u'] = len(self.G.adj[u])

        starttime = datetime.datetime.now()
        for u in self.rank:
            value = self.check_SCANA_core(u)
            if self.G.nodes[u]['l'] >= self.miu:
                self.add_node_set(u)
                self.cluster_SCANA_core(u)

        endtime = datetime.datetime.now()
        interval = (endtime - starttime).total_seconds()
        #print("Runing time of SCANA:" + str(interval))
        self.write_runtime(interval, sys._getframe().f_code.co_name)
        self.sigma = {}
        self.sigma_t = {}
        self.visited_node = []
        file_name = self.path + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANA'
        #print("Cores output at: " + file_name)
        file_object = open(file_name, 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()
        self.union_set = []

    def frquent_mining(self, u):
        candidate_set = []
        for v in self.G.adj[u]:
            if len(self.G.edges[u, v]['t']) >= self.tau:
                candidate_set.append(v)
        if len(candidate_set) < self.miu:
            return False

        candidate_time = {}
        for v in candidate_set:
            for time_item in self.G.edges[u, v]['t']:
                if time_item in candidate_time:
                    candidate_time[time_item] += 1
                else:
                    candidate_time[time_item] = 1

        times_more_than_miu = []
        for key in candidate_time:
            if candidate_time[key] >= self.miu:
                times_more_than_miu.append(key)
        if len(times_more_than_miu) < self.tau:
            return False

        listtemp = itertools.combinations(times_more_than_miu, self.tau)
        time_set = list(listtemp)
        for set_temp in time_set:
            frquent_vertices = []
            for time_item in set_temp:
                vertices_at_time_item = []
                for v in candidate_set:
                    if time_item not in self.G.edges[u, v]['t']:
                        continue

                    if v >= u:
                        edge_set = (u, v, time_item)
                    else:
                        edge_set = (v, u, time_item)

                    if edge_set not in self.sigma_t:
                        self.sigma_t[edge_set] = self.compute_sigma_at_one_time(u, v, time_item)

                    if self.sigma_t[edge_set] > self.eps:
                        vertices_at_time_item.append(v)

                if len(vertices_at_time_item) >= self.miu:
                    frquent_vertices.append(vertices_at_time_item)
                else:
                    break

            if len(frquent_vertices) < self.tau:
                break

            set_result = frquent_vertices[0]
            for i in range(1, len(frquent_vertices)):
                set_result = set(frquent_vertices[i]) & set(set_result)

            if len(set_result) >= self.miu:
                # print(u,set_temp,set_result)
                # self.frquent_set = {}
                return True

        return False

    def write_runtime(self, t, module_name):
        file_object = open('running time', 'a')
        time = {"name": self.path, "eps": self.eps, "tau": self.tau, "miu": self.miu, "time": t,
                "method": module_name}
        file_object.write(json.dumps(time))
        file_object.write("\n")
        file_object.close()

    def cluster_by_cores(self, file_c, flag):
        self.union_set = []
        nodes_set = []

        if flag == 1:
            for line in open(file_c):
                number = int(line.strip('\n'))
                nodes_set.append(number)
        if flag == 2:
            for line in open(file_c):
                new_dict = json.loads(line)
                for item in new_dict:
                    nodes_set.append(item)

        for u in nodes_set:
            self.add_node_set(u)
            for v in set(nodes_set) & set(self.G.adj[u]):
                if v > u:
                    self.add_node_set(v)
                    result = self.compute_sigma(u, v)
                    if result >= self.tau:
                        self.union(u, v)

        cluster_ans = []
        for i in range(len(self.union_set)):
            cluster1 = self.union_set[i][:]
            for u in self.union_set[i]:
                for v in set(self.G.adj[u]):
                    if v >= u:
                        edge_set = (u, v)
                    else:
                        edge_set = (v, u)
                    if edge_set not in self.sigma:
                        result = self.compute_sigma(u, v)
                        self.sigma[edge_set] = result
                    if self.sigma[edge_set] >= self.tau:
                        cluster1.append(v)
            cluster_ans.append(set(cluster1))
        # pprint.pprint(self.union_set)
        file_object = open(file_c + '_cluster', 'w')
        for unit in cluster_ans:
            file_object.write(json.dumps(list(unit)))
            file_object.write("\n")
        file_object.close()

        self.union_set = []
        self.sigma = {}
        self.sigma_t = {}
        self.visited_node = []

    def run(self, filename):
        self.SCANB(self.miu, self.tau, self.eps)
        file_object = open(filename + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANB', 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()

        self.SCANB_slow(self.miu, self.tau, self.eps)
        file_object = open(filename + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANB-', 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()

        self.SCANW(self.miu, self.tau, self.eps)
        file_object = open(filename + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANW', 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()

        self.SCANS(self.miu, self.tau, self.eps)
        file_object = open(filename + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANS', 'w')
        for unit in self.union_set:
            file_object.write(json.dumps(unit))
            file_object.write("\n")
        file_object.close()

    def cluster(self, filename):
        newname = filename + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANB'
        self.cluster_by_cores(newname, 2)
        newname = filename + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANA'
        self.cluster_by_cores(newname, 2)
        newname = filename + '.output-' + str(self.eps) + '-' + str(self.tau) + '-' + str(self.miu) + '_SCANS'
        self.cluster_by_cores(newname, 1)

    def evauluation(self, filename, clustername):
        name1 = filename + '.output-' + clustername + '_SCANB_cluster'
        name2 = filename + '.output-' + clustername + '_SCANS_cluster'
        name3 = filename + '.output-' + clustername + '_SCANA_cluster'
        
        s1 = self.separability(filename, name1)
        s2 = self.separability(filename, name2)
        s3 = self.separability(filename, name3)
        # print("separability", s1, s2, s3)

        d1 = self.density(filename, name1)
        d2 = self.density(filename, name2)
        d3 = self.density(filename, name3)
        # print("density", d1, d2, d3)

        c1 = self.cohesiveness(filename, name1)
        c2 = self.cohesiveness(filename, name2)
        c3 = self.cohesiveness(filename, name3)
        # print("cohesiveness", c1, c2, c3)
        

        finalDict = {"SCANB": [s1, d1, c1], "SCANS": [s2, d2, c2], "SCANA": [s3, d3, c3]}
        nameJSON = filename + '.output-' + clustername + '_metrics.json'
        with open(nameJSON, 'w') as f:
            json.dump(finalDict, f, indent=4)

        # G.ccoefficient(filename, name1)
        # G.ccoefficient(filename, name2)
        # G.ccoefficient(filename, name3)

    def evaluaition_by_year(self, filename, clustername):
        name1 = filename + '.output-' + clustername + '_SCANB_cluster'
        name2 = filename + '.output-' + clustername + '_SCANS_cluster'
        name3 = filename + '.output-' + clustername + '_SCANA_cluster'
        G.separability_by_year(filename, name1)
        # G.separability_by_year(filename, name2)
        # G.separability_by_year(filename, name3)
        G.density_by_year(filename, name1)
        G.density_by_year(filename, name2)
        G.density_by_year(filename, name3)
        G.cohesiveness_by_year(filename, name1)
        # G.cohesiveness_by_year(filename, name2)
        # G.cohesiveness_by_year(filename, name3)
        G.ccoefficient_by_year(filename, name1)
        G.ccoefficient_by_year(filename, name2)
        G.ccoefficient_by_year(filename, name3)

    def degree_distribution_by_year(self):
        degree = {}
        for t in range(100):
            degree_t = 0
            for u in self.rank:
                for v in self.adj[u]:
                    if t in self.G.edges[u, v]['t']:
                        degree_t += 1
            degree[t] = degree_t
        #print(list(degree.keys()))
        #print(list(degree.values()))

    def nodes_distribution_by_year(self):
        degree = {}
        for t in range(100):
            degree_t = 0
            for u in self.rank:
                for v in self.adj[u]:
                    if t in self.G.edges[u, v]['t']:
                        degree_t += 1
                        break
            degree[t] = degree_t
        #print(list(degree.keys()))
        #print(list(degree.values()))

    def degree_distribution_of_nodes(self):
        degree = {}
        for u in self.rank:
            degree_t = 0
            for v in self.adj[u]:
                degree_t += len(self.G.edges[u, v]['t'])
            if degree_t in degree:
                degree[degree_t] += 1
            else:
                degree[degree_t] = 1
        #print(list(degree.keys()))
        #print(list(degree.values()))

    def degree_distribution_of_nodes_detemporal(self):
        degree = {}
        for u in self.rank:
            degree_t = len(self.adj[u])
            if degree_t in degree:
                degree[degree_t] += 1
            else:
                degree[degree_t] = 1
        #print(list(degree.keys()))
        #print(list(degree.values()))


def run_algorithm(G, eps, tau, miu):
    G.SCANB(miu, tau, eps)
    G.SCANA(miu, tau, eps)
    G.SCANS(miu, tau, eps)

def compute_metrics(target):
    data_dir = "data"

    # Paràmetres a evaluar
    eps_list = [0.2, 0.5, 0.8]
    tau_list = [3, 5, 10]
    miu_list = [3, 5, 10]

    default_eps = 0.2
    default_tau = 3
    default_miu = 3

    for root, _, files in (os.walk(data_dir)):
        # try:
        #     if root.split("\\")[-2] in []:
        #         continue
        #     if root.split("\\")[1] == "ELDP":
        #         continue
        # except:
        #     pass

        for file in tqdm(files, desc="Processing files in root: " + str(root)):

            if not (file.endswith('.txt')) or not (file.startswith(target)):
                continue
                
            filepath = os.path.join(root, file)
            try:
                G = tGraph(filepath)
            except:
                continue
            for eps in eps_list:
                try:
                    run_algorithm(G, eps, default_tau, default_miu)
                    G.cluster(filepath)
                    clustername = str(eps) + '-' + str(default_tau) + '-' + str(default_miu)
                    G.evauluation(filepath, clustername)
                except:
                    continue
            for tau in tau_list:
                try:
                    run_algorithm(G, default_eps, tau, default_miu)
                    G.cluster(filepath)
                    clustername = str(default_eps) + '-' + str(tau) + '-' + str(default_miu)
                    G.evauluation(filepath, clustername)
                except:
                    continue
            for miu in miu_list:
                try:
                    run_algorithm(G, default_eps, default_tau, miu)
                    G.cluster(filepath)
                    clustername = str(default_eps) + '-' + str(default_tau) + '-' + str(miu)
                    G.evauluation(filepath, clustername)
                    # G.separability(datapath, clusterpath) # Evaluació de si les comunitats estan ben separades de les altres de la xarxa
                    # G.density(datapath, clusterpath) # Evaluació Connectivitat entre nodes d'una comunitat
                    # G.cohesiveness(datapath, clusterpath) # Evaluació estructura interna de la comunitat (ha de ser difícil separar la comunitat entre dos sub-comunitats)
                except:
                    continue

def view_results(target):
    data_dir = "data"
    dictResults = defaultdict(lambda: defaultdict(dict))  # dict[group][param_value][file]

    def get_group(root):
        if "ELDP" in root:
            return "ELDP"
        elif "KDA" in root:
            return "KDA"
        else:
            return "original_graphs"

    def extract_main_param(file_name, group):
        match = re.search(r'_([\d\.]+)\.txt', file_name)
        if match:
            return match.group(1)  # ej: "0.1" para ELDP
        return "unknown"

    for root, _, files in os.walk(data_dir):
        group = get_group(root)
        for file in files:
            if not file.endswith('.json') or not file.startswith(target):
                continue
            
            if "0.8-3-3" in file:
                continue

            filepath = os.path.join(root, file)
            try:
                with open(filepath) as f:
                    data = json.load(f)
            except:
                continue

            if any(data[scan] == [None, None, 0] for scan in ['SCANA', 'SCANB', 'SCANS']):
                continue

            param_value = extract_main_param(file, group)
            if file not in dictResults[group][param_value]:
                dictResults[group][param_value][file] = {'SCANB': [], 'SCANA': [], 'SCANS': []}

            for scan in ['SCANB', 'SCANA', 'SCANS']:
                dictResults[group][param_value][file][scan].append(data[scan])

    # Configuración gráfica
    variables = ['AS', 'AD', 'AC']
    scans = ['SCANA', 'SCANB', 'SCANS']
    colors = {'SCANA': 'tab:blue', 'SCANB': 'tab:green', 'SCANS': 'tab:orange'}
    offset = {'SCANA': -0.2, 'SCANB': 0, 'SCANS': 0.2}
    width = 0.2
    
    for group, param_dict in dictResults.items():
        param_values = sorted(param_dict.keys(), key=lambda x: float(x) if x != "unknown" else float('inf'))
        n_params = len(param_values)

        
        fig, axes = plt.subplots(n_params, 3, figsize=(16, 2 * n_params), sharex=False)
        

        if n_params == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, param_value in enumerate(param_values):
            files_data = param_dict[param_value]
            dictResults2 = {}
            for key, value in files_data.items():
                if all(len(value[scan]) == 0 for scan in scans):
                    continue
                arraySCANB = np.array(value['SCANB'])
                arraySCANA = np.array(value['SCANA'])
                arraySCANS = np.array(value['SCANS'])

                dictResults2[key] = {
                    'SCANB': [np.mean(arraySCANB, axis=0).tolist(), np.std(arraySCANB, axis=0).tolist()],
                    'SCANA': [np.mean(arraySCANA, axis=0).tolist(), np.std(arraySCANA, axis=0).tolist()],
                    'SCANS': [np.mean(arraySCANS, axis=0).tolist(), np.std(arraySCANS, axis=0).tolist()]
                }

            if not dictResults2:
                continue
            files = list(dictResults2.keys())
            y = np.arange(len(files))

            def extract_params(file_name):
                match = re.search(r'output-([\d\.]+)-([\d]+)-([\d]+)', file_name)
                if match:
                    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                return file_name

            for var_idx, var_name in enumerate(variables):
                ax = axes[row_idx, var_idx]
                for scan in scans:
                    x_pos = y + offset[scan]
                    y_vals = [dictResults2[f][scan][0][var_idx] for f in files]
                    std_vals = [dictResults2[f][scan][1][var_idx] for f in files]
                    ax.bar(x_pos, y_vals, width=width, label=scan, color=colors[scan], yerr=std_vals, capsize=5)

                if group == "ELDP":
                    parameter = r"$\epsilon$"
                    ax.set_title(f'{var_name}, {parameter} = {param_value}', fontsize=10)
                elif group == "KDA":
                    parameter = "k"
                    ax.set_title(f'{var_name}, {parameter} = {param_value}', fontsize=10)
                else:
                    ax.set_title(f'{var_name}', fontsize=15)

                ax.set_xticks(y)
                ax.set_xticklabels([extract_params(f) for f in files], fontsize=8, ha='right')
                if row_idx == n_params - 1:
                    ax.set_xlabel('Paràmetres (ε-τ-μ)', fontsize=12)
                ax.set_ylabel(var_name, fontsize=12)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)


        fig.suptitle(f'Mètriques de detecció de comunitats en {target} \n {group}, #Experiments = 5', fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()  



if __name__ == '__main__':
    FILES = ['mammalia-voles', 'insecta-ant', 
             'WEEK_ia-enron', 'MONTH_ia-enron',
             'WEEK_CollegeMsg', 'MONTH_CollegeMsg']
    
    TARGET_FILE = FILES[-1]
    compute_metrics(TARGET_FILE)
    view_results(TARGET_FILE)

    # filename = "chess_year"
    # G = tGraph(filename)
    # G.eps = 0.5
    # G.tau = 3
    # G.miu = 3

    # G.SCANB(G.miu, G.tau, G.eps)
    # G.SCANA(G.miu, G.tau, G.eps)
    # G.SCANS(G.miu, G.tau, G.eps)

    # G.cluster(filename)
    # clustername = str(G.eps) + '-' + str(G.tau) + '-' + str(G.miu)
    # G.evauluation(filename, clustername)