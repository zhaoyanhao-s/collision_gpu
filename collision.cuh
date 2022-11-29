#ifndef COLLISION_CUH
#define COLLISION_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include "Simpson.h"
#include <vector>
#include <iostream>
#include <stdint.h>
#include <tuple>
#include <unordered_map>

#define POINTS_NUM 400000
#define ACCURACY   10000
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

class gVector {
public:
    float m_x;
    float m_y;
    float m_z;
public:
    __device__ __host__ float& operator[](const int& i);
    __device__ __host__ gVector& operator=(const gVector& rhs);
    __device__ __host__ gVector(const float& x, const float& y, const float& z) :
        m_x(x), m_y(y), m_z(z) {
    }
    __device__ __host__ gVector() = default;
};

class gPoint {
public:
	float m_x;
	float m_y;
	float m_z;
	gPoint* next;
public:
	__device__ __host__ float& operator[](const int& i);
    __device__ __host__ gPoint(const gVector& rhs) :m_x(rhs.m_x), m_y(rhs.m_y), m_z(rhs.m_z), next(nullptr) {};
	__device__ __host__ gPoint& operator=(const gPoint& rhs);
	__device__ __host__ gPoint(const float& x, const float& y, const float& z, gPoint* pPoint = nullptr) :
		m_x(x), m_y(y), m_z(z), next(pPoint) {
	}
	__device__ __host__ gPoint() = default;

    __device__ __host__ double norm() {
        return sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
    }
};



class gCube {
public:
	float m_halflength{ 0.0f };
	//长方体的中心点
	gPoint m_center{ 0.0f,0.0f,0.0f };
	__device__  __host__ gCube() {};
	__device__  __host__ gCube(const float& halflength, const gPoint& center) :
		m_halflength(halflength),
		m_center(center) {
	};
	__device__  __host__  ~gCube() {};
	__device__ __host__ float operator[](int i);
	__device__  __host__ float operator[](int i) const;
};
class gCuboid {
public:
    float m_halflength{ 0.0f };
    float m_halfwidth{ 0.0f };
    float m_halfheight{ 0.0f };
    gVector m_axis[3] = {
        {1,0,0},
        {0,1,0},
        {0,0,1}
    };
    //长方体的中心点
    gPoint m_center{ 0.0f,0.0f,0.0f };
    __device__ __host__ gCuboid() {};
    __device__ __host__ gCuboid(const float& halflength, const float& halfwidth, const float& halfheight, const gPoint& center) :
        m_halflength(halflength),
        m_halfwidth(halfwidth),
        m_halfheight(halfheight),
        m_center(center) {
    };
    __device__ __host__ void setAxe(const float& roation);
    __device__ __host__ float operator[](int i);
    __device__ __host__ float operator[](int i) const;
};
class gOctreeNode {
public:
    __device__ __host__ gOctreeNode(const gPoint& center, const float& halflength);
    __device__ __host__ gOctreeNode() = default;
    __device__ __host__ ~gOctreeNode();
public:
    float m_halflength{ 0.0f };
    gPoint m_center{ 0.0f,0.0f,0.0f };
    gOctreeNode* m_parent{ nullptr };
    gOctreeNode* m_children[8];
    gPoint* m_scenepointslist{ nullptr };
    gPoint* m_gpupointlist{ nullptr };
    unsigned int m_pointnum = 0;
    uint8_t  ChildExists = 0; // optional
    uint64_t MortanCode = 1;
};

using GPU_GPUNode = std::tuple<gOctreeNode*, gOctreeNode*>;
class gOctree
{
public:
    __device__ __host__ gOctree();
    __device__ __host__ gOctreeNode* cpuGetRoot();
    __device__ __host__ gOctreeNode* gpuGetRoot();
    gOctreeNode* cpuLookupNode(uint64_t locCode);
    gOctreeNode* gpuLookupNode(uint64_t locCode);
    GPU_GPUNode BuildOctree(gOctreeNode* parent, uint64_t MortanCode, const gPoint& center, const float& halflength, int stopDepth);
    void InsertObject(gPoint* cpoint,gPoint* gpoint, uint64_t MortanCode = 1);
    
public:
    //GPUNodes&CPUNodes都存储在CPU上，只是方便host方便访问GPU的接口
    std::unordered_map<uint64_t, gOctreeNode*> CPUNodes;
    std::unordered_map<uint64_t, gOctreeNode*> GPUNodes;
};
class Octree
{
public:
    __device__ static void gpuInsertObject(gPoint* gpoint, gOctreeNode* gOctree);
    __device__ static gOctreeNode* gpuBuildOctree(gOctreeNode* parent, uint64_t MortanCode, const gPoint& center, const float& halflength, int stopDepth);
};

template<class T>
__device__ __host__ T gAbs(const T& x);
__device__ __host__ bool abortTesting(const float& znodedown, const float& zrobotup, const float& znodeup, const float& zrobotdown);
__device__ __host__ bool isCollisionOBB2D(const gOctreeNode& box1, const gCuboid& box2);
__device__ __host__ bool isSeparatingLine(const gOctreeNode& box1, const gCuboid& box2, const gVector& axis, const gVector& delta);
__device__ bool getCollisionPointGPU(const gOctreeNode* collisionnode, const gCuboid& robot);
__host__ bool getCollisionPointCPU(const gOctreeNode* collisionnode, const gCuboid& robot);
__device__ bool getCollisionOctreeNodeOBB2DGPU(gOctreeNode* node, const gCuboid& robot);
__host__ bool getCollisionOctreeNodeOBB2DCPU(gOctreeNode* node, const gCuboid& robot);
__device__ __host__ bool isCollisionProject(gPoint* p, const gCuboid& robot);

__device__ __host__ gPoint operator+(const gPoint& lhs, const gVector& rhs);
__device__ __host__ gPoint operator+(const gVector& lhs, const gPoint& rhs);
__device__ __host__ gPoint operator+(const gPoint& lhs, const gPoint& rhs);
__device__ __host__ gVector operator-(const gVector& lhs, const gVector& rhs);
__device__ __host__ gPoint operator-(const gPoint& lhs, const gVector& rhs);
__device__ __host__ gVector operator-(const gPoint& lhs, const gPoint& rhs);
__device__ __host__ float operator* (const gVector& lhs, const gVector& rhs);
__device__ __host__ gVector operator* (const gVector& lhs, const float& rhs);
__device__ __host__ gPoint operator* (const gPoint& lhs, const float& rhs);
class BezierCurve3 {
public:
    gPoint p0, p1, p2, p3;
    BezierCurve3(const gPoint& a, const gPoint& b, const gPoint& c, const gPoint& d) {
        p0 = a;
        p1 = b;
        p2 = c;
        p3 = d;
    }
    BezierCurve3() = default;
    gPoint at(float t) const {       //返回函数值
        return p0 * (pow(1 - t, 3)) + p1 * 3 * t * pow(1 - t, 2) + p2 * 3 * pow(t, 2) * (1 - t) + p3 * pow(t, 3);
    }
    gPoint dev(float t) const {      //一阶导数 对参数t
        gPoint k1 = (p3 - p2 * 3 + p1 * 3 - p0) * 3;
        gPoint k2 = (p2 - p1 * 2 + p0) * 6;
        gPoint k3 = (p1 - p0) * 3;
        return k1 * pow(t, 2) + k2 * t + k3;
    }
    gPoint dev2(float t) const {     //二阶导数 对参数t
        gPoint k1 = (p3 - p2 * 3 + p1 * 3 - p0) * 6;
        gPoint k2 = (p2 - p1 * 2 + p0) * 6;
        return k1 * t + k2;
    }
    float total_length() const {
        auto df = [&](float t) -> float
        {
            return this->dev(t).norm();
        };
        return NumericalIntegration::adaptive_simpson_3_8(df, 0, 1);
    }
    float length_with_t(float t) const {
        auto df = [&](float t) -> float
        {
            return this->dev(t).norm();
        };
        return NumericalIntegration::adaptive_simpson_3_8(df, 0, t);
    }
    float getTByArcLength_Steffensen(float start, float target_length) const {
        size_t max_iter_time = 10;
        float iter_eps = 0.0001;
        auto df = [&](double t) -> float
        {
            return this->dev(t).norm();
        };
        float approx_t = start + target_length / total_length();
        //std::cout << "approx_t_1 = " << approx_t << std::endl;
        float b = NumericalIntegration::adaptive_simpson_3_8(df, start, approx_t) - target_length;
        float k = 2 * b * this->dev(approx_t).norm();
        if (k != 0)
            approx_t = -b / k + approx_t;
        //std::cout << "approx_t_2 = " << approx_t << std::endl;
        float prev_approx_t = 0;
        if (approx_t > 1 || approx_t < 0) {
        }

        for (int iter = 0; iter < max_iter_time; ++iter)
        {
            float approx_length = NumericalIntegration::adaptive_simpson_3_8(df, start, approx_t);
            float d = approx_length - target_length;
            if (abs(d) < iter_eps) {
                return approx_t;
            }

            // Newton's method
            float first_order = this->dev(approx_t).norm();
            float numerator = pow(d, 2);
            float denominator = 2 * d * first_order;
            float temp_k1 = approx_t - numerator / denominator;

            // Newton's method
            approx_length = NumericalIntegration::adaptive_simpson_3_8(df, start, temp_k1);
            d = approx_length - target_length;
            if (abs(d) < iter_eps) {
                return approx_t;
            }
            first_order = this->dev(temp_k1).norm();
            numerator = pow(d, 2);
            denominator = 2 * d * first_order;
            float temp_k2 = temp_k1 - numerator / denominator;

            //Steffensen accelerate
            numerator = (temp_k1 - approx_t) * (temp_k1 - approx_t);
            denominator = temp_k2 - 2 * temp_k1 + approx_t;
            approx_t = approx_t - numerator / denominator;
            if (abs(approx_t - prev_approx_t) < iter_eps) {
                //std::cout << "The number of iterations is " << iter << std::endl;
                return approx_t;
            }
            else prev_approx_t = approx_t;
        }
        return approx_t;
    }
    float getCurvature(const double& t) { //当曲率小于一定值时，认为沿直线前进
        float yz = pow(dev2(t).m_z * dev(t).m_y - dev(t).m_z * dev2(t).m_y, 2);
        float xz = pow(dev2(t).m_x * dev(t).m_z - dev(t).m_x * dev2(t).m_z, 2);
        float xy = pow(dev2(t).m_y * dev(t).m_x - dev(t).m_y * dev2(t).m_x, 2);

        float numerator = sqrt(yz + xy + xz);
        float denominator = pow(dev(t).m_x * dev(t).m_x + dev(t).m_y * dev(t).m_y + dev(t).m_z * dev(t).m_z, 1.5);
        if (denominator < 1e-6)
            return -1; //表示此处曲率无穷大
        else
            return numerator / denominator;
    }
};

__global__ void calculateCollision(gOctreeNode* octree, gCuboid* robotpart, int* res);
__global__ void visitOctreeNode(gOctreeNode* node);
__global__ void buildOctree(gOctreeNode* d_root);

void callCollision_4_256(gOctreeNode* octree, gCuboid* robotpart, int* res);
void callCollision_8_128(gOctreeNode* octree, gCuboid* robotpart, int* res);
void callCollision_16_64(gOctreeNode* octree, gCuboid* robotpart, int* res);

void callCollision(gOctreeNode* octree, gCuboid* robotpart, int* res);
void callvisitOctreeNode(gOctreeNode* node);
void callBuildOctree(gOctreeNode* d_root);
#endif // !COLLISION_CUH