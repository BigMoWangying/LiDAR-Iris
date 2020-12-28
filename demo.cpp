#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/kdtree/kdtree_flann.h>
#include "LidarIris.h"

using namespace std;

// number of sequence
const int N = 2761;

// kitti sequence
const string seq = "05";

/*0 for kitti "00","05" only same direction loops;
1 for kitti "08" only reverse loops; 
2 for both same and reverse loops*/
const int loop_event = 0;

std::vector<vector<int>> getGTFromPose(const string& pose_path)
{
    std::ifstream pose_ifs(pose_path);
    std::string line;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    int index = 1;
    while(getline(pose_ifs, line)) 
    {
        if(line.empty()) break;
        stringstream ss(line);
        float r1,r2,r3,t1,r4,r5,r6,t2,r7,r8,r9,t3;
        ss >> r1 >> r2 >> r3 >> t1 >> r4 >> r5 >> r6 >> t2 >> r7 >> r8 >> r9 >> t3;
        pcl::PointXYZI p;
        p.x = t1;
        p.y = 0;
        p.z = t3;
        p.intensity = index++;
        cout << p << endl;
        cloud->push_back(p);
    }

    pcl::io::savePCDFileASCII(seq +".pcd", *cloud);
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud);
    std::vector<vector<int>> res(5000);
    for(int i = 0; i < cloud->points.size(); i++)
    {
        float radius = 4;
        std::vector<int> ixi;
        std::vector<float> ixf;
        pcl::PointXYZI p = cloud->points[i];
        int cur = p.intensity;
        std::vector<int> nrs;
        kdtree.radiusSearch(p,radius,ixi,ixf);
        for(int j = 0; j < ixi.size(); j++)
        {
            if(cloud->points[ixi[j]].intensity == cur) continue;
            nrs.push_back(cloud->points[ixi[j]].intensity);
        }
        sort(nrs.begin(), nrs.end());
        res[cur] = nrs;
    }
    
    std::ofstream gt_ofs("../gt"+ seq + ".txt");

    for(int i =0; i < res.size(); i++)
    {
        gt_ofs << i << " ";
        for(int j = 0; j < res[i].size(); j++)
        {
            gt_ofs << res[i][j] << " ";
        }
        gt_ofs << endl;
    }
    return res;
}

int main(int argc, char *argv[])
{

    //kitti pose xx.txt
    auto gt = getGTFromPose("/media/yingwang/Document/" +seq + "/"+seq +".txt");
    std::ofstream ofs("../test_res" + seq+".txt");

    LidarIris iris(4, 18, 1.6, 0.75, loop_event);
    std::vector<LidarIris::FeatureDesc> dataset(N);
    for(int i =0; i <=N-1 ;i++)
    {
        std::stringstream ss;
        ss << setw(6) << setfill('0') << i;
        cout << ss.str()+".bin" << std::endl;

        // kitti velodyne bins
        std::string filename = "/media/yingwang/Document/" + seq + "/velodyne/" + ss.str() + ".bin";
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>);
        std::fstream input(filename, std::ios::in | std::ios::binary);
        input.seekg(0, std::ios::beg);
        for (int ii=0; input.good() && !input.eof(); ii++) {
            pcl::PointXYZ point;
            input.read((char *) &point.x, 3*sizeof(float));
            float intensity;
            input.read((char *) &intensity, sizeof(float));
            cloud0->push_back(point);
        }
        cv::Mat1b li1 = LidarIris::GetIris(*cloud0);
        LidarIris::FeatureDesc fd1 = iris.GetFeature(li1);
        dataset[i] = fd1;
        float mindis = 1000;
        int loop_id = -1;
        for(int j = 0; j <= i-300; j++)
        {
            LidarIris::FeatureDesc fd2 = dataset[j];

            int bias;
            auto dis = iris.Compare(fd1, fd2, &bias);
            if(dis < mindis)
            {
                mindis = dis;
                loop_id = j;
            }


        }

        if(loop_id == -1) continue;
        if(std::find(gt[i+1].begin(),gt[i+1].end(),loop_id+1)!=gt[i+1].end())
        {
            ofs << i+1 << " " << loop_id+1 << " " << mindis << " " << 1 << std::endl; 
            cout << i+1 << " " << loop_id+1 << " " << mindis << " " << 1 << std::endl; 
        }
        else 
        {
            ofs << i+1 << " " << loop_id+1 << " " << mindis << " " << 0 << std::endl; 
            cout << i+1 << " " << loop_id+1 << " " << mindis << " " << 0 << std::endl; 

        }


    }

    return 0;
}