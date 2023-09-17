#ifndef KDTREE_YW_H
#define KDTREE_YW_H

#include <algorithm>
#include <cassert>
#include <vector>
#include <queue>

template <typename T>
class KDTree
{
private:
    struct Node
    {
        Node(int k);
        ~Node();

        bool is_leaf();

        T *point;
        int index;
        int splitDim;

        Node *left;
        Node *right;
    };

    struct NodeCompare
    {
        NodeCompare(int dim);
        bool operator()(const Node *n1, const Node *n2);

        int dim;
    };

    struct PQEntry
    {
        PQEntry(Node *node, T dist) : node(node), dist(dist){};

        Node *node;
        T dist;
    };

    struct PQCompare
    {
        bool operator()(const PQEntry &e1, const PQEntry &e2);
    };

public:
    KDTree(){};
    KDTree(int k, T *points, int n);
    ~KDTree();

    void init(int k, T *points, int n);
    void nn(T *point, int *outputPtr);
    void knn(int k, T *point, int *output, int *outputCountPtr);

private:
    Node *build_tree(Node **start, Node **end, int splitDim);
    void nn_internal(T *point, Node *node, Node **nearestNodePtr, T *nearestDistPtr);
    void knn_internal(int k, T *point, Node *node, std::priority_queue<PQEntry, std::vector<PQEntry>, PQCompare> &pq);
    T compute_node_point_dist(const Node *node, T *point);

    Node *root;
    int nDim;
};

// Node Definition
/////////////////////////////////////////////////////////////////////////////////

template <typename T>
KDTree<T>::Node::Node(int k)
{
    this->point = new T[k];
    this->left = nullptr;
    this->right = nullptr;
}

template <typename T>
KDTree<T>::Node::~Node()
{
    delete[] this->point;

    if (this->left)
        delete this->left;
    if (this->right)
        delete this->right;
}

template <typename T>
bool KDTree<T>::Node::is_leaf()
{
    return !this->left && !this->right;
}

// NodeCompare Definition
/////////////////////////////////////////////////////////////////////////////////

template <typename T>
KDTree<T>::NodeCompare::NodeCompare(int dim)
{
    this->dim = dim;
}

template <typename T>
bool KDTree<T>::NodeCompare::operator()(const Node *n1, const Node *n2)
{
    return n1->point[this->dim] < n2->point[this->dim];
}

// PQCompare Definition
/////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool KDTree<T>::PQCompare::operator()(const PQEntry &e1, const PQEntry &e2)
{
    return e1.dist < e2.dist;
}

// KDTree Definition
/////////////////////////////////////////////////////////////////////////////////

template <typename T>
KDTree<T>::KDTree(int k, T *points, int n)
{
    this->init(k, points, n);
}

template <typename T>
KDTree<T>::~KDTree()
{
    if (this->root)
    {
        delete this->root;
    }
}

template <typename T>
void KDTree<T>::init(int k, T *points, int n)
{
    assert((void("k should be greater than 0"), k > 0));

    this->nDim = k;

    Node **nodes = new Node *[n];

    for (int i = 0; i < n; ++i)
    {
        nodes[i] = new Node(k);

        std::copy(points + i * k, points + (i + 1) * k, &nodes[i]->point[0]);
        nodes[i]->index = i;
    }

    this->root = this->build_tree(nodes, nodes + n, 0);

    delete[] nodes;
}

template <typename T>
void KDTree<T>::nn(T *point, int *outputPtr)
{
    Node *nearestNode;
    T nearestNodeDist;

    this->nn_internal(point, this->root, &nearestNode, &nearestNodeDist);

    *outputPtr = nearestNode->index;
}

template <typename T>
void KDTree<T>::knn(int k, T *point, int *output, int *outputCountPtr)
{
    *outputCountPtr = 0;

    if (k > 0)
    {
        std::priority_queue<PQEntry, std::vector<PQEntry>, PQCompare> pq;
        this->knn_internal(k, point, this->root, pq);

        for (int i = 0; i < k && pq.size(); ++i)
        {
            output[k - i - 1] = pq.top().node->index;
            *outputCountPtr += 1;
            pq.pop();
        }
    }
}

template <typename T>
typename KDTree<T>::Node *KDTree<T>::build_tree(Node **start, Node **end, int splitDim)
{
    int n = end - start;
    int nextDim = (splitDim + 1) % this->nDim;

    if (n == 0)
    {
        return nullptr;
    }

    std::nth_element(start, start + n / 2, end, NodeCompare(splitDim));

    Node *node = start[n / 2];
    node->splitDim = splitDim;
    node->left = this->build_tree(start, start + n / 2, nextDim);
    node->right = this->build_tree(start + n / 2 + 1, end, nextDim);

    return node;
}

template <typename T>
void KDTree<T>::nn_internal(T *point, Node *node, Node **nearestNodePtr, T *nearestDistPtr)
{
    if (!node)
        return;

    bool isLeaf = node->is_leaf();
    int splitDim = node->splitDim;

    bool visitedLeft;

    Node *nearestChild;
    T nearestChildDist;

    if (!isLeaf)
    {
        if (point[splitDim] < node->point[splitDim])
        {
            this->nn_internal(point, node->left, &nearestChild, &nearestChildDist);
            visitedLeft = true;
        }
        else
        {
            this->nn_internal(point, node->right, &nearestChild, &nearestChildDist);
            visitedLeft = false;
        }
    }

    T dist = this->compute_node_point_dist(node, point);

    if (dist < nearestChildDist)
    {
        nearestChild = node;
        nearestChildDist = dist;
    }

    if (nearestChildDist > abs(point[splitDim] - node->point[splitDim]))
    {
        Node *otherNearestChild;
        T otherNearestChildDist;

        if (visitedLeft)
        {
            this->nn_internal(point, node->right, &otherNearestChild, &otherNearestChildDist);
        }
        else
        {
            this->nn_internal(point, node->left, &otherNearestChild, &otherNearestChildDist);
        }

        if (otherNearestChildDist < nearestChildDist)
        {
            nearestChild = otherNearestChild;
            nearestChildDist = otherNearestChildDist;
        }
    }

    *nearestNodePtr = nearestChild;
    *nearestDistPtr = nearestChildDist;
}

template <typename T>
void KDTree<T>::knn_internal(int k, T *point, Node *node, std::priority_queue<PQEntry, std::vector<PQEntry>, PQCompare> &pq)
{
    if (!node)
        return;

    bool isLeaf = node->is_leaf();
    int splitDim = node->splitDim;

    bool visitedLeft;

    pq.push(PQEntry(node, this->compute_node_point_dist(node, point)));

    if (pq.size() > k)
    {
        pq.pop();
    }

    if (!isLeaf)
    {
        if (point[splitDim] < node->point[splitDim])
        {
            this->knn_internal(k, point, node->left, pq);
            visitedLeft = true;
        }
        else
        {
            this->knn_internal(k, point, node->right, pq);
            visitedLeft = false;
        }
    }

    T maxDist = pq.top().dist;

    if (maxDist > abs(point[splitDim] - node->point[splitDim]))
    {
        if (visitedLeft)
        {
            this->knn_internal(k, point, node->right, pq);
        }
        else
        {
            this->knn_internal(k, point, node->left, pq);
        }
    }
}

template <typename T>
T KDTree<T>::compute_node_point_dist(const Node *node, T *point)
{
    T sum = 0;

    for (int i = 0; i < this->nDim; ++i)
    {
        T diff = node->point[i] - point[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}

#endif