#include "collision.cuh"
#include "mytimer.h"



__device__ __host__ void getPath(const int& accuracy,std::vector<gPoint>& m_path,std::vector<float>& m_rotation,const BezierCurve3& beziercurve) {
    m_path.clear();
    m_rotation.clear();
    float start = 0;
    float target_length = beziercurve.total_length() / accuracy;
    for (int i = 0; i <= accuracy; ++i) {
        float t = beziercurve.getTByArcLength_Steffensen(start * i, target_length * (i + 1));
        gVector dev1 = { beziercurve.dev(t).m_x,beziercurve.dev(t).m_y,beziercurve.dev(t).m_z };
        float rotation = atan(dev1.m_y / dev1.m_x);
        gPoint pathPoint = beziercurve.p0 * (pow(1 - t, 3))
            + beziercurve.p1 * 3 * t * pow(1 - t, 2)
            + beziercurve.p2 * 3 * pow(t, 2) * (1 - t)
            + beziercurve.p3 * pow(t, 3);
        m_path.push_back(pathPoint);
        m_rotation.push_back(rotation);
    }
}


gCuboid setRobotPos(gCuboid& robot,const gPoint& newcenter, const float& newrotation) {
    robot.m_center.m_x = newcenter.m_x;
    robot.m_center.m_y = newcenter.m_y;
    robot.setAxe(newrotation);
    return robot;
}


int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    gPoint* host_point;
    gCuboid* host_robotpart;
    int* host_res;
    host_point = (gPoint*)malloc(sizeof(gPoint) * POINTS_NUM);
    host_robotpart = (gCuboid*)malloc(sizeof(gCuboid));
    host_res = (int*)malloc(sizeof(int) * ACCURACY);

    //gOctreeNode* d_r;
    //callBuildOctree(d_r);

    host_robotpart->m_center = gPoint(0, 0, 0);
    host_robotpart->m_halfheight = 0.6f;
    host_robotpart->m_halflength = 0.2f;
    host_robotpart->m_halfwidth = 0.375f;
    host_robotpart->m_axis[0] = gVector(1, 0, 0);
    host_robotpart->m_axis[1] = gVector(0, 1, 0);

    BezierCurve3 beziercurve;
    beziercurve.p0 = host_robotpart->m_center + gVector(0, 0, 0);
    beziercurve.p1 = host_robotpart->m_center + gVector(10, 0, -0.375);
    beziercurve.p2 = host_robotpart->m_center + gVector(-10, -20, -0.375);
    beziercurve.p3 = host_robotpart->m_center + gVector(-30, -45, -0.375);
    std::vector<gPoint> m_path;
    std::vector<float> m_rotation;
    getPath(ACCURACY, m_path,m_rotation, beziercurve);


    srand((unsigned)time(NULL));
    for (int i = 0; i < POINTS_NUM; ++i)
        {
            int length = 50 * 200;
            host_point[i].m_x = static_cast<float>(rand() % length - 50*100)/100;
            host_point[i].m_y = static_cast<float>(rand() % length - 50*100)/100;
            host_point[i].m_z = static_cast<float>(rand() % length - 50*100)/100;
        }

    //将路径上的机器人姿态拷贝到设备
    gCuboid* h_robot_position = (gCuboid*)malloc(sizeof(gCuboid) * ACCURACY);
    gCuboid* d_robot_position;
    CELLTimestamp time0;
    float timer0 = 0;
    
    CHECK(cudaMalloc((void**)&d_robot_position, sizeof(gCuboid) * ACCURACY));
    time0.update();
    for (int i = 0; i < ACCURACY; ++i)
    {
        h_robot_position[i] = setRobotPos(*host_robotpart, m_path[i], m_rotation[i]);
    }
    
    CHECK(cudaMemcpy(d_robot_position, h_robot_position, sizeof(gCuboid) * ACCURACY, cudaMemcpyHostToDevice));
    timer0 += time0.getElapsedTimeInMilliSec();
    std::cout << "Data transform has consumed time = " << timer0 << "ms"<<std::endl;

    gPoint* device_point;
    gCuboid* device_robotpart;
    int* device_res;
    CHECK(cudaMalloc((void**)&device_point, sizeof(gPoint)* POINTS_NUM));
    CHECK(cudaMalloc((void**)&device_robotpart, sizeof(gCuboid)));
    CHECK(cudaMalloc((void**)&device_res, sizeof(int) * ACCURACY));
    CHECK(cudaMemcpy(device_point, host_point, sizeof(gPoint)* POINTS_NUM, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_robotpart, host_robotpart, sizeof(gCuboid), cudaMemcpyHostToDevice));


    std::cout << "Loading the octree...Wait patiently\n";
    gOctree* octree = new gOctree();
    octree->BuildOctree(nullptr, 1, gPoint(0, 0, 0), 50, 0);
    std::cout << "Loading completed\n";
    std::cout << "Resolution is 0.02m\n";
    std::cout << "The octree occupied space is 50m * 50m * 50m\n\n"; 

    std::cout << "Inserting points to octree...Wait patiently\n";
    for (int i = 0; i < POINTS_NUM; ++i) {
        octree->InsertObject(&host_point[i], &device_point[i]);
    }
    std::cout << "Loading completed\n\n";
    std::cout << "Total number of load points is "<< POINTS_NUM<<"\n";
    std::cout << "The number of tests is " << ACCURACY << "\n";

    gOctreeNode* root = octree->gpuGetRoot();
    CELLTimestamp TIME;
    float time = 0.f;
    
    TIME.update();
    callCollision(root, d_robot_position, device_res);
    CHECK(cudaMemcpy(host_res, device_res, sizeof(int) * ACCURACY, cudaMemcpyDeviceToHost));
    time += TIME.getElapsedTimeInMilliSec();
    std::cout << "GPU<<<16,1024>>> TIME = " << time <<"ms" << std::endl;
    //std::cout << "GPU RES==================================================================\n";
    //for (int i = 0; i < 1000; ++i)
    //    std::cout << host_res[i] << "  ";


    time = 0.f;
    TIME.update();
    callCollision_4_256(root, d_robot_position, device_res);
    CHECK(cudaMemcpy(host_res, device_res, sizeof(int) * ACCURACY, cudaMemcpyDeviceToHost));
    time += TIME.getElapsedTimeInMilliSec();
    std::cout << "GPU<<<64,256>>>  TIME = " << time << "ms" << std::endl;

    time = 0.f;
    TIME.update();
    callCollision_8_128(root, d_robot_position, device_res);
    CHECK(cudaMemcpy(host_res, device_res, sizeof(int) * ACCURACY, cudaMemcpyDeviceToHost));
    time += TIME.getElapsedTimeInMilliSec();
    std::cout << "GPU<<<128,128>>>  TIME = " << time << "ms" << std::endl;

    time = 0.f;
    TIME.update();
    callCollision_16_64(root, d_robot_position, device_res);
    CHECK(cudaMemcpy(host_res, device_res, sizeof(int) * ACCURACY, cudaMemcpyDeviceToHost));
    time += TIME.getElapsedTimeInMilliSec();
    std::cout << "GPU<<<256,64>>>  TIME = " << time << "ms" << std::endl;

    //for (auto gnode : octree->GPUNodes) {
    //    callvisitOctreeNode(gnode.second);
    //}



   
    gOctreeNode* croot = octree->cpuGetRoot();
    time = 0.f;
    TIME.update();
    for (int i = 0; i < ACCURACY; ++i) {
        getCollisionOctreeNodeOBB2DCPU(croot, h_robot_position[i]);
    }
    time += TIME.getElapsedTimeInMilliSec();
    //std::cout << "CPU RES==================================================================\n";
    //for (int i = 0; i < ACCURACY; ++i)
    //    std::cout << host_res[i] << "  ";
    std::cout << "CPU<<<1,1>>>    TIME = " << time << "ms" << std::endl;


    
    time = 0.f;
    TIME.update();
    int *real_host_res = (int*)malloc(sizeof(int) * ACCURACY);
    for (int i = 0; i < ACCURACY; ++i) {
        bool res = false;
        for (int j = 0; j < POINTS_NUM; ++j) {
            if (isCollisionProject(&host_point[j], h_robot_position[i]))
            res = true;
        }
        real_host_res[i] = res;
    }
    time += TIME.getElapsedTimeInMilliSec();
    std::cout << "BAOLI   TIME = " << time << "ms" << std::endl;
    //std::cout << "REAL RES==================================================================\n";


    bool res = false;
    for (int i = 0; i < ACCURACY; ++i) {
        if (real_host_res[i] != host_res[i])
        {
            printf("%d\n", i);
            res = true;
        }
    }
    if (res == false) {
        printf("GPU alogrithm is right\n");
    }
    else
        printf("GPU alogrithm is error\n");


    cudaFree(device_point);
    cudaFree(device_robotpart);
    cudaFree(device_res);

    free(host_point);
    free(host_res);
    free(host_robotpart);
}