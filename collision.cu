#include "collision.cuh"
__device__ __host__ float& gPoint::operator[](const int& i) {
    switch (i) {
    case 0:
        return m_x;
    case 1:
        return m_y;
    case 2:
        return m_z;
    default:
        break;
    }
}
__device__ __host__ gPoint& gPoint::operator=(const gPoint& rhs) {
    if (&rhs != this) {
        m_x = rhs.m_x;
        m_y = rhs.m_y;
        m_z = rhs.m_z;
        next = rhs.next;
    }
    return *this;
}
__device__ __host__ float& gVector::operator[](const int& i) {
    switch (i) {
    case 0:
        return m_x;
    case 1:
        return m_y;
    case 2:
        return m_z;
    default:
        break;
    }
}
__device__ __host__ gVector& gVector::operator=(const gVector& rhs) {
    if (&rhs != this) {
        m_x = rhs.m_x;
        m_y = rhs.m_y;
        m_z = rhs.m_z;
    }
    return *this;
}
__device__ __host__ float gCube::operator[](int i) {
    float b[3] = { m_halflength,m_halflength,m_halflength };
    return b[i];
}
__device__ __host__ float gCube::operator[](int i) const {
    float b[3] = { m_halflength,m_halflength,m_halflength };
    return b[i];
}

__device__ __host__ void gCuboid::setAxe(const float& rotation) {
    m_axis[0].m_x = m_axis[1].m_y = cos(rotation);
    m_axis[0].m_y = sin(rotation);
    m_axis[1].m_x = -sin(rotation);
}

__device__ __host__ float gCuboid::operator[](int i) {
    float b[3] = { m_halflength,m_halfwidth,m_halfheight };
    return b[i];
}
__device__ __host__ float gCuboid::operator[](int i) const {
    float b[3] = { m_halflength,m_halfwidth,m_halfheight };
    return b[i];
}
__device__ __host__ gOctreeNode::gOctreeNode(const gPoint& center, const float& halflength)
{
    m_center = center;
    m_halflength = halflength;
    m_scenepointslist = nullptr;
    m_gpupointlist = nullptr;
    m_pointnum = 0;
    ChildExists = 0;
    MortanCode = 1;
    for (int i = 0; i < 8; ++i) {
        m_children[i] = nullptr;
    }
};
gOctree::gOctree() {
    CPUNodes.clear();
    GPUNodes.clear();
}
gOctreeNode* gOctree::cpuGetRoot() {
    return cpuLookupNode(1);
}
gOctreeNode* gOctree::gpuGetRoot() {
    return gpuLookupNode(1);
}

//std::get<0> == host_node
//std::get<1> == device_node
GPU_GPUNode gOctree::BuildOctree(gOctreeNode* parent, uint64_t MortanCode, const gPoint& center, const float& halflength, int stopDepth) {
    if (stopDepth < 0) {
        return GPU_GPUNode(nullptr, nullptr);
    }
    else {
        gOctreeNode* host_node = new gOctreeNode();
        gOctreeNode* device_node;
        CHECK(cudaMalloc((void**)&device_node, sizeof(gOctreeNode)));
        host_node->m_center = center;
        host_node->m_halflength = halflength;
        host_node->m_scenepointslist = nullptr;
        host_node->m_gpupointlist = nullptr;
        host_node->m_pointnum = 0;
        host_node->m_parent = parent;
        host_node->MortanCode = MortanCode;
        CPUNodes[MortanCode] = host_node;
        GPUNodes[MortanCode] = device_node;
        for (int i = 0; i < 8; ++i) {
            host_node->m_children[i] = nullptr;
        }
        gVector offset;
        MortanCode <<= 3;
        float lengthstep = halflength * 0.5f;
        gOctreeNode* CPUchildren[8] = { nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr };
        if (--stopDepth >= 0) {
            for (int i = 0; i < 8; ++i) {
                uint64_t MortanCode_copy = MortanCode;
                offset.m_x = ((i & 1) ? lengthstep : -lengthstep);
                offset.m_y = ((i & 2) ? lengthstep : -lengthstep);
                offset.m_z = ((i & 4) ? lengthstep : -lengthstep);
                MortanCode_copy = MortanCode | i;
                gPoint realcenter = center + offset;
                GPU_GPUNode gcnode = BuildOctree(host_node, MortanCode_copy, realcenter, lengthstep, stopDepth);
                host_node->m_children[i] = std::get<1>(gcnode);
                CPUchildren[i] = std::get<0>(gcnode);
                host_node->ChildExists |= (1 << i);
            }
        }
        CHECK(cudaMemcpy(device_node, host_node, sizeof(gOctreeNode), cudaMemcpyHostToDevice));
        for (int i = 0; i < 8; ++i) {
            host_node->m_children[i] = CPUchildren[i];
        }
        return { host_node,device_node };
    }
}

__device__ gOctreeNode* Octree::gpuBuildOctree(gOctreeNode* parent, uint64_t MortanCode, const gPoint& center, const float& halflength, int stopDepth) {
    if (stopDepth < 0) {
        return nullptr;
    }
    else {
        gOctreeNode* device_node;
        cudaMalloc((void**)&device_node, sizeof(gOctreeNode));
        device_node->m_center = center;
        device_node->m_halflength = halflength;
        device_node->m_scenepointslist = nullptr;
        device_node->m_gpupointlist = nullptr;
        device_node->m_pointnum = 0;
        device_node->m_parent = parent;
        device_node->MortanCode = MortanCode;
        //printf("Node halflength = %f\n", device_node->m_halflength);
        gVector offset;
        float lengthstep = halflength * 0.5f;
        if (--stopDepth >= 0) {
            for (int i = 0; i < 8; ++i) {
                uint64_t MortanCode_copy = MortanCode;
                offset.m_x = ((i & 1) ? lengthstep : -lengthstep);
                offset.m_y = ((i & 2) ? lengthstep : -lengthstep);
                offset.m_z = ((i & 4) ? lengthstep : -lengthstep);
                gPoint realcenter = center + offset;
                device_node->ChildExists |= (1 << i);
                device_node->m_children[i] =  gpuBuildOctree(device_node, MortanCode_copy, realcenter, lengthstep, stopDepth);
            }
        }
        return device_node;
    }
}

void gOctree::InsertObject(gPoint* cpoint, gPoint* gpoint, uint64_t MortanCode) {
    gOctreeNode* device_node = gpuLookupNode(MortanCode);
    gOctreeNode* host_node = cpuLookupNode(MortanCode);
    if (cpoint == nullptr || host_node == nullptr) {
        std::cout << "Insert nullptr\n";
        return;
    }

    int index = 0;
    for (int i = 0; i < 3; i++)
    {
        float delta = (*cpoint)[i] - host_node->m_center[i];
        if (delta > 0.0f)
            index |= (1 << i);
    }
    MortanCode <<= 3;
    if (host_node->ChildExists & (1 << index)) {
        MortanCode |= index;
        InsertObject(cpoint, gpoint ,MortanCode);
    }
    else if (host_node->m_scenepointslist != nullptr && host_node->m_halflength >= 0.02) {
        gVector offset;
        float lengthstep = host_node->m_halflength * 0.5f;
        gOctreeNode* CPUchildren[8];
        for (int i = 0; i < 8; ++i) {
            uint64_t MortanCode_copy = MortanCode;
            offset.m_x = ((i & 1) ? lengthstep : -lengthstep);
            offset.m_y = ((i & 2) ? lengthstep : -lengthstep);
            offset.m_z = ((i & 4) ? lengthstep : -lengthstep);
            gPoint realcenter = host_node->m_center + offset;
            MortanCode_copy = MortanCode | i;

            gOctreeNode* device_child;
            CHECK(cudaMalloc((void**)&(device_child), sizeof(gOctreeNode)));
            CPUchildren[i] = new gOctreeNode(realcenter, lengthstep);
            host_node->m_children[i] = device_child; //方便复制到device
            host_node->ChildExists |= (1 << i);

            CPUchildren[i]->m_parent = device_node;
            CPUchildren[i]->MortanCode = MortanCode_copy;

            CPUNodes[MortanCode_copy] = CPUchildren[i];
            GPUNodes[MortanCode_copy] = device_child;
            CHECK(cudaMemcpy(device_child, CPUchildren[i], sizeof(gOctreeNode), cudaMemcpyHostToDevice));
        }

        uint64_t MortanCodeParent = MortanCode;
        MortanCode |= index;
        InsertObject(cpoint,gpoint, MortanCode);
        if (host_node->m_scenepointslist) {
            int idx = 0;
            for (int i = 0; i < 3; i++)
            {
                float delta = (*host_node->m_scenepointslist)[i] - host_node->m_center[i];
                if (delta > 0.0f)
                    idx |= (1 << i);
            }
            MortanCodeParent |= idx;
            InsertObject(host_node->m_scenepointslist,host_node->m_gpupointlist, MortanCodeParent);
            host_node->m_scenepointslist = nullptr;
            host_node->m_gpupointlist = nullptr;
        }
        CHECK(cudaMemcpy(device_node, host_node, sizeof(gOctreeNode), cudaMemcpyHostToDevice));
        for (int i = 0; i < 8; ++i) {
            host_node->m_children[i] = CPUchildren[i];
            CPUchildren[i]->m_parent = host_node;
        }
    }
    else {
        cpoint->next = host_node->m_gpupointlist;
        CHECK(cudaMemcpy(gpoint,cpoint, sizeof(gPoint), cudaMemcpyHostToDevice));
        cpoint->next = host_node->m_scenepointslist;
        host_node->m_scenepointslist = cpoint;
        host_node->m_gpupointlist = gpoint;
        CHECK(cudaMemcpy(device_node, host_node, sizeof(gOctreeNode), cudaMemcpyHostToDevice));
    }
}

__device__ void Octree::gpuInsertObject(gPoint* gpoint, gOctreeNode* gOctree) {
    if (gpoint == nullptr || gOctree == nullptr) {
        printf("Insert nullptr\n");
        return;
    }

    int index = 0;
    for (int i = 0; i < 3; i++)
    {
        float delta = (*gpoint)[i] - gOctree->m_center[i];
        if (delta > 0.0f)
            index |= (1 << i);
    }
    if (gOctree->ChildExists & (1 << index)) {
        gpuInsertObject(gpoint, gOctree->m_children[index]);
    }
    else if (gOctree->m_scenepointslist != nullptr && gOctree->m_halflength >= 0.02) {
        gVector offset;
        float lengthstep = gOctree->m_halflength * 0.5f;
        for (int i = 0; i < 8; ++i) {
            offset.m_x = ((i & 1) ? lengthstep : -lengthstep);
            offset.m_y = ((i & 2) ? lengthstep : -lengthstep);
            offset.m_z = ((i & 4) ? lengthstep : -lengthstep);
            gPoint realcenter = gOctree->m_center + offset;

            gOctreeNode* device_child;
            cudaMalloc((void**)&(device_child), sizeof(gOctreeNode));
            gOctree->m_children[i] = device_child; //方便复制到device
            gOctree->ChildExists |= (1 << i);

            device_child->m_parent = gOctree;
        }

        gpuInsertObject(gpoint, gOctree->m_children[index]);
        if (gOctree->m_gpupointlist) {
            int idx = 0;
            for (int i = 0; i < 3; i++)
            {
                float delta = (*gOctree->m_gpupointlist)[i] - gOctree->m_center[i];
                if (delta > 0.0f)
                    idx |= (1 << i);
            }
            gpuInsertObject(gOctree->m_gpupointlist, gOctree->m_children[idx]);
            gOctree->m_gpupointlist = nullptr;
        }
    }
    else {
        gpoint->next = gOctree->m_gpupointlist;
        gOctree->m_gpupointlist = gpoint;
    }
}
gOctreeNode* gOctree::cpuLookupNode(uint64_t locCode)
{
    const auto iter = CPUNodes.find(locCode);
    return (iter == CPUNodes.end() ? nullptr : (*iter).second);
}
gOctreeNode* gOctree::gpuLookupNode(uint64_t locCode)
{
    const auto iter = GPUNodes.find(locCode);
    return (iter == GPUNodes.end() ? nullptr : (*iter).second);
}
//
__device__ __host__ gOctreeNode::~gOctreeNode() {
    while (m_scenepointslist != nullptr) {
        gPoint* next = m_scenepointslist->next;
        delete m_scenepointslist;
        m_scenepointslist = next;
    }
    if (m_children[0] != nullptr) {
        for (int i = 0; i < 8; ++i) {
            delete m_children[i];
        }
    }
}

template<class T>
__device__ __host__ T gAbs(const T& x) {
    return x > 0 ? x : -x;
}
__device__ __host__ bool abortTesting(const float& znodedown, const float& zrobotup, const float& znodeup, const float& zrobotdown) {
    return (znodedown >= zrobotup || znodeup <= zrobotdown);
}
__device__ __host__ bool isSeparatingLine(const gOctreeNode& box1, const gCuboid& box2, const gVector& axis, const gVector& delta) {
    return box1.m_halflength * gAbs(axis * gVector(1,0,0)) + box1.m_halflength * gAbs(axis * gVector(0, 1, 0))\
        + box2.m_halflength * gAbs(axis * box2.m_axis[0]) + box2.m_halfwidth * gAbs(axis * box2.m_axis[1])\
        <= gAbs(delta * axis);
}
//通过OBB包围盒判断碰撞 2D
__device__ __host__ bool isCollisionOBB2D(const gOctreeNode& box1, const gCuboid& box2) {
    gVector delta = box1.m_center - box2.m_center;
    if (isSeparatingLine(box1, box2, gVector(1, 0, 0), delta)) return false;
    if (isSeparatingLine(box1, box2, gVector(0, 1, 0), delta)) return false;
    if (isSeparatingLine(box1, box2, box2.m_axis[0], delta)) return false;
    if (isSeparatingLine(box1, box2, box2.m_axis[1], delta)) return false;
    return true;
}
__device__  bool getCollisionPointGPU(const gOctreeNode* collisionnode, const gCuboid& robot) {
    gPoint* realhead = collisionnode->m_gpupointlist;
    while (realhead != nullptr) {
        if (isCollisionProject(realhead, robot)) {
            return true;
        }
        realhead = realhead->next;
    }
    return false;
}
__host__  bool getCollisionPointCPU(const gOctreeNode* collisionnode, const gCuboid& robot) {
    gPoint* realhead = collisionnode->m_scenepointslist;
    while (realhead != nullptr) {
        if (isCollisionProject(realhead, robot)) {
            return true;
        }
        realhead = realhead->next;
    }
    return false;
}
__host__ bool getCollisionOctreeNodeOBB2DCPU(gOctreeNode* node, const gCuboid& robot) {
    if (node == nullptr)
        return false;

    if (abortTesting(node->m_center[2] - node->m_halflength, robot.m_center.m_z + robot.m_halfheight, \
        node->m_center[2] + node->m_halflength, robot.m_center.m_z - robot.m_halfheight))
        return false;

    if (isCollisionOBB2D(*node, robot)) {
        if (node->ChildExists != 0) {
            bool res = false;
            for (int i = 0; i < 8; ++i) {
                if (getCollisionOctreeNodeOBB2DCPU(node->m_children[i], robot))
                    res = true;
            }
            return res;
        }
        else if (getCollisionPointCPU(node, robot)) {
            return true;
        }
        else
            return false;
    }
    else
        return false;
}
__device__ bool getCollisionOctreeNodeOBB2DGPU(gOctreeNode* node, const gCuboid& robot) {
    if (node == nullptr)
        return false;

    if (abortTesting(node->m_center[2] - node->m_halflength, robot.m_center.m_z + robot.m_halfheight, \
        node->m_center[2] + node->m_halflength, robot.m_center.m_z - robot.m_halfheight)) 
        return false;

    if (isCollisionOBB2D(*node, robot)) {
        if (node->m_children[0] != 0) {
            for (int i = 0; i < 8; ++i) {
                if (getCollisionOctreeNodeOBB2DGPU(node->m_children[i], robot))
                    return true;
            }
            return false;
        }
        else if (getCollisionPointGPU(node, robot)) {
            return true;
        }
        else
            return false;
    }
    else
        return false;
}
__device__ __host__ bool isCollisionProject(gPoint* p, const gCuboid& robot) {
    if (p->m_z > robot.m_center.m_z + robot.m_halfheight || \
        p->m_z < robot.m_center.m_z - robot.m_halfheight)
        return false;

    gVector vec1 = (*p) - robot.m_center;

    float proj_y = gAbs(vec1 * robot.m_axis[1]);
    float proj_x = gAbs(vec1 * robot.m_axis[0]);

    if (proj_x <= robot.m_halflength && proj_y <= robot.m_halfwidth) {
        return true;
    }
    else
        return false;
}

__device__ __host__ gPoint operator+(const gPoint& lhs, const gVector& rhs) {
    return { lhs.m_x + rhs.m_x, lhs.m_y + rhs.m_y, lhs.m_z + rhs.m_z };
}
__device__ __host__ gPoint operator+(const gPoint& lhs, const gPoint& rhs) {
    return { lhs.m_x + rhs.m_x, lhs.m_y + rhs.m_y, lhs.m_z + rhs.m_z };
}
__device__ __host__ gPoint operator+(const gVector& lhs, const gPoint& rhs) {
    return { lhs.m_x + rhs.m_x, lhs.m_y + rhs.m_y, lhs.m_z + rhs.m_z };
}
__device__ __host__ gVector operator-(const gVector& lhs, const gVector& rhs) {
    return gVector{ lhs.m_x - rhs.m_x, lhs.m_y - rhs.m_y, lhs.m_z - rhs.m_z };
}
__device__ __host__ gPoint operator-(const gPoint& lhs, const gVector& rhs) {
    return gPoint{ lhs.m_x - rhs.m_x, lhs.m_y - rhs.m_y, lhs.m_z - rhs.m_z };
}
__device__ __host__ gVector operator-(const gPoint& lhs, const gPoint& rhs) {
    return gVector{ lhs.m_x - rhs.m_x, lhs.m_y - rhs.m_y, lhs.m_z - rhs.m_z };
}
__device__ __host__ float operator* (const gVector& lhs, const gVector& rhs) {
    return lhs.m_x * rhs.m_x + lhs.m_y * rhs.m_y + lhs.m_z * rhs.m_z;
}

__device__ __host__ gVector operator* (const gVector& lhs, const float& rhs) {
    return gVector{ lhs.m_x * rhs, lhs.m_y * rhs, 0 };
}
__device__ __host__ gPoint operator* (const gPoint& lhs, const float& rhs) {
    return gPoint{ lhs.m_x * rhs, lhs.m_y * rhs, 0 };
}
 __global__ void calculateCollision(gOctreeNode* octree,gCuboid* robotpart,int* res) {
     int i = threadIdx.x + blockIdx.x * blockDim.x;
     if (i < ACCURACY) {
         res[i] = getCollisionOctreeNodeOBB2DGPU(octree, robotpart[i]);
     }
}

 __global__ void visitOctreeNode(gOctreeNode* node) {
     if (node->m_gpupointlist == nullptr) {
         //printf("Empty\n");
     }
     else {
         printf("Node's address is %p\n", node);
         printf("Node's halflength is %f\n", node->m_halflength);
         printf("Node's center x = %f, y = %f, z = %f\n", node->m_center.m_x, node->m_center.m_y, node->m_center.m_z);
         printf("Node's children's address is %p\n", node->m_children);
     }
 }
 __global__ void buildOctree(gOctreeNode* d_root) {
     d_root = Octree::gpuBuildOctree(nullptr, 1, gPoint(0, 0, 0), 50, 5);
 }
void callCollision(gOctreeNode* octree, gCuboid* robotpart, int* res) {
	calculateCollision << <16,1024 >> > (octree, robotpart, res);
	cudaDeviceSynchronize();
}
void callCollision_4_256(gOctreeNode* octree, gCuboid* robotpart, int* res) {
    calculateCollision << <64, 256 >> > (octree, robotpart, res);
    cudaDeviceSynchronize();
}
void callCollision_8_128(gOctreeNode* octree, gCuboid* robotpart, int* res) {
    calculateCollision << <128, 128 >> > (octree, robotpart, res);
    cudaDeviceSynchronize();
}
void callCollision_16_64(gOctreeNode* octree, gCuboid* robotpart, int* res) {
    calculateCollision << <256, 64 >> > (octree, robotpart, res);
    cudaDeviceSynchronize();
}
void callvisitOctreeNode(gOctreeNode* node) {
    visitOctreeNode << <1, 1 >> > (node);
    cudaDeviceSynchronize();
}
void callBuildOctree(gOctreeNode* d_root) {
    buildOctree << <1, 1 >> > (d_root);
}