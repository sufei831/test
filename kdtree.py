import numpy as np
from  math import sqrt
from sklearn.datasets import load_iris

class KdNode(object):
    def __init__(self,dom_elt,split,left,right):
        self.dom_elt=dom_elt
        self.split=split
        self.left=left
        self.right=right

class KdTree(object):
    def  __init__(self,distance):
        self.root=None
        self.distance=distance

    def create(self,data):
        k=len(data[0])
        def _CreateNode(split,data_set):
            if not data_set:  #递归出口
                return None
        # 排序
            data_set.sort(key=lambda x:x[split])
            split_pos=len(data_set)//2
            median=data_set[split_pos]  #中位数节点
            split_next=(split+1)%k     #下一个节点
            return KdNode(
                median,
                split,
                _CreateNode(split_next,data_set[:split_pos]),  #创建左子树
                _CreateNode(split_next,data_set[split_pos+1:])  #创建右子树
            )
        self.root=_CreateNode(0,data)
    def print(self):
        def _peorder(node:KdNode):
            print(node.dom_elt)
            if node.left:
                _peorder(node.left)
            else:
                print("#")
            if node.right:
                _peorder(node.right)
            else:
                 print("#")
        print("Peorder:")
        _peorder(self.root)
        print("="*10)
    def find_nn(self,point):
        def _search(start_node:KdNode):
            print("#go search")
            if start_node:print("start at:",start_node.dom_elt)
            while start_node:
                stack.append(start_node)
                if point[start_node.split] <= start_node.dom_elt[start_node.split]:
                    start_node=start_node.left
                    if start_node: print("left", start_node.dom_elt)
                else:
                    start_node = start_node.right
                    if start_node:print("right",start_node.dom_elt)
            print("#end search")

        min_node,min_dist=None,float("inf")
        if self.root is None:
            return min_node,min_dist
        stack=[]
        _search(self.root)
        nodes_visited=0
        while stack:
            node=stack.pop()
            print("back to:",node.dom_elt)
            dist=self.distance(node.dom_elt,point)
            nodes_visited+=1
            if dist <min_dist:
                min_dist=dist
                min_node=node
                print("update min_dist",min_node.dom_elt,min_dist)
            if node.left is None and node.right is None:
                continue
            else:
                offset=abs(node.dom_elt[node.split]-point[node.split])
                if offset < min_dist:
                    if(point[node.split]>node.dom_elt[node.split])  and node.left:
                        print("check overlapping on the other side(left):",node.left.dom_elt)
                        _search(node.left)
                    elif(point[node.split]<=node.dom_elt[node.split])  and node.right:
                        print("check overlapping on the other side(right):", node.right.dom_elt)
                        _search(node.right)
        print("nodes_visited:",nodes_visited)
        return min_node,min_dist
def test(T:KdTree,point):
    min_node,min_dist=T.find_nn(point)
    print(min_node.dom_elt,min_dist)

if __name__=="__main__":
    data=[[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    kd_tree=KdTree(lambda x,y:sqrt(sum((p1-p2)**2 for p1,p2 in zip(x,y))))
    kd_tree.create(data)
    kd_tree.print()

    point=[2.1,3.1]
    test(kd_tree,point)

    print("brute-forde checking:")
    dists=[sqrt(sum((p1-p2)**2 for p1,p2 in zip(point,d)))for d in data]
    min_dist=min(dists)
    min_idx=dists.index(min_dist)
    print(data[min_idx],min_dist)














