#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

// Portable, checksum-pinned collision support geometry shared with Python.
class CollisionPlaneModel {
    using Json=nlohmann::json;
    struct Joint {std::string name,parent,child,type;Eigen::Matrix4d origin;Eigen::Vector3d axis;};
    std::vector<Joint> joints;
    std::map<std::string,std::vector<Eigen::Vector3d>> vertices;
    std::set<std::string> moving;
 public:
    void load(const std::string &path) {
        std::ifstream file(path,std::ios::binary);
        if(!file)throw std::runtime_error("Cannot open plane-only URDF collision model: "+path);
        std::string raw((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
        unsigned char digest[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(raw.data()),raw.size(),digest);
        std::ostringstream hex;for(unsigned char c:digest)hex<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<int>(c);
        if(hex.str()!="92acf86aa21f99c19cfb970e2caeb3ecbda13ebba5885dd4464dbd7c9fbdd5d7")throw std::runtime_error("Unreviewed URDF collision model checksum");
        Json data=Json::parse(raw);
        for(const auto &link:data.at("links")){
            auto &v=vertices[link.at("name").get<std::string>()];
            for(const auto &point:link.at("vertices"))v.emplace_back(point[0].get<double>(),point[1].get<double>(),point[2].get<double>());
        }
        for(const auto &entry:data.at("joints")){
            Joint j;j.name=entry.at("name");j.parent=entry.at("parent");j.child=entry.at("child");j.type=entry.at("type");
            for(int r=0;r<4;++r)for(int c=0;c<4;++c)j.origin(r,c)=entry.at("origin")[r][c].get<double>();
            j.axis=Eigen::Vector3d::Zero();
            if(j.type!="fixed")for(int i=0;i<3;++i)j.axis[i]=entry.at("axis")[i].get<double>();
            joints.push_back(j);
        }
        std::set<std::string> resolved={"base_link"};
        size_t previous=0;
        while(resolved.size()!=vertices.size()){
            previous=resolved.size();
            for(const auto &j:joints)if(resolved.count(j.parent)&&!resolved.count(j.child)){
                if(j.type!="fixed"||moving.count(j.parent))moving.insert(j.child);
                resolved.insert(j.child);
            }
            if(previous==resolved.size())throw std::runtime_error("Invalid collision model tree");
        }
    }
    void check(const std::vector<double> &q,const std::vector<double> &hand,const std::vector<double> &tcp,
               const Eigen::Vector3d &normal,double offset,double required) const {
        if(q.size()!=6||hand.size()!=20||tcp.size()!=7||vertices.empty())throw std::runtime_error("Missing plane-only surface feedback/model");
        const char *arm[]={"shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"};
        std::map<std::string,double> angles;
        for(int i=0;i<6;++i)angles[arm[i]]=q[i];
        for(int f=1;f<=5;++f)for(int j=1;j<=4;++j)angles["rj_dg_"+std::to_string(f)+"_"+std::to_string(j)]=hand[(f-1)*4+j-1];
        std::map<std::string,Eigen::Matrix4d> frames;frames["base_link"]=Eigen::Matrix4d::Identity();
        while(frames.size()!=vertices.size()){
            size_t previous=frames.size();
            for(const auto &j:joints)if(frames.count(j.parent)&&!frames.count(j.child)){
                Eigen::Matrix4d local=j.origin;
                if(j.type!="fixed")local.block<3,3>(0,0)=j.origin.block<3,3>(0,0)*Eigen::AngleAxisd(angles.at(j.name),j.axis.normalized()).toRotationMatrix();
                frames[j.child]=frames.at(j.parent)*local;
            }
            if(previous==frames.size())throw std::runtime_error("Unresolved collision frames");
        }
        Eigen::Matrix4d base_from_root=frames.at("base").inverse();
        for(const auto &link:vertices){
            if(!moving.count(link.first))continue; // Invariant mounting base cannot be moved by calibration.
            Eigen::Matrix4d t=base_from_root*frames.at(link.first);
            Eigen::Vector3d local_normal=t.block<3,3>(0,0).transpose()*normal;
            const double bias=normal.dot(t.block<3,1>(0,3))+offset;
            for(const auto &point:link.second)if(local_normal.dot(point)+bias<required)
                throw std::runtime_error("Plane-only surface clearance violated: "+link.first);
        }
        Eigen::Quaterniond rotation(tcp[6],tcp[3],tcp[4],tcp[5]);
        Eigen::Vector3d translation(tcp[0],tcp[1],tcp[2]);
        for(double x:{-.046,.046})for(double z:{-.046,.046})
            if(normal.dot(rotation*Eigen::Vector3d(x,-.05,z)+translation)+offset<required)
                throw std::runtime_error("Plane-only wrist-board clearance violated");
    }
};
