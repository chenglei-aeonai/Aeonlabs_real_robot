#include "calibration_guard.hpp"
#include <fstream>
#include <iostream>
#include <functional>

void expect_failure(const std::function<void()> &action) {
    bool failed=false;
    try { action(); } catch(const std::exception &) { failed=true; }
    if(!failed) throw std::runtime_error("Expected guard refusal");
}

int main(int argc,char **argv) {
    if(argc<2||argc>3) return 2;
    nlohmann::json data;
    std::ifstream input(argv[1]); input>>data;
    CalibrationGuard guard(data);
    const std::vector<double> pose={0,0,1,0,0,0,1}, joints={0,0,0,0,0,0}, zero(6,0);
    guard.check_pose(pose); guard.check_arm(joints);
    expect_failure([&]{auto bad=data;bad["verified"]=false;CalibrationGuard rejected(bad);});
    expect_failure([&]{auto bad=data;bad["tool_radius_m"]=nullptr;CalibrationGuard rejected(bad);});
    expect_failure([&]{auto bad=data;bad["max_linear_speed_m_s"]=1;CalibrationGuard rejected(bad);});
    expect_failure([&]{auto p=pose;p[0]=2.9;guard.check_pose(p);});
    expect_failure([&]{auto p=pose;p[2]=-1.9;guard.check_pose(p);});
    expect_failure([&]{auto bad=data;bad["table_top_z_m"]=.2;CalibrationGuard collision(bad);collision.check_arm(joints);});
    if(guard.status().at("version").get<int>()!=5) throw std::runtime_error("Wrong plane-guard version");
    expect_failure([&]{auto bad=data;bad.erase("table_plane");CalibrationGuard rejected(bad);});
    expect_failure([&]{auto bad=data;bad["table_plane"]["frame"]="camera";CalibrationGuard rejected(bad);});
    expect_failure([&]{auto bad=data;bad["table_plane"]["normal"]={0,0,-1};CalibrationGuard rejected(bad);});
    expect_failure([&]{auto bad=data;bad["table_plane"]["normal"]={0,0,2};CalibrationGuard rejected(bad);});
    expect_failure([&]{auto p=pose;p[2]=-1.8;guard.check_pose(p);});
    auto sloped=data;
    sloped["table_plane"]["normal"]={-std::sin(CalibrationGuard::radians(10)),0.,std::cos(CalibrationGuard::radians(10))};
    sloped["table_top_z_m"]=-1.6;
    CalibrationGuard tilted_plane(sloped);
    tilted_plane.check_pose(pose);
    expect_failure([&]{sloped["table_top_z_m"]=-2.;CalibrationGuard rejected(sloped);});
    const auto now=CalibrationGuard::Clock::now();
    expect_failure([&]{guard.target("unarmed",pose,now);});
    guard.arm("test",pose,joints);
    guard.arm("test",pose,joints); // acknowledgement retry must not change anchor
    expect_failure([&]{guard.arm("other",pose,joints);});
    auto below=pose;below[2]-=.01;
    expect_failure([&]{guard.target("test",below,now);});
    auto downward_settings=data;downward_settings["max_downward_offset_m"]=.05;
    CalibrationGuard downward(downward_settings);
    downward.arm("down",pose,joints);
    auto permitted_down=pose;permitted_down[2]-=.05;
    downward.target("down",permitted_down,now);
    expect_failure([&]{auto p=permitted_down;p[2]-=.001;downward.target("down",p,now);});
    expect_failure([&]{auto invalid=data;invalid["max_downward_offset_m"]=.051;CalibrationGuard refused(invalid);});
    auto relaxed_settings=data;relaxed_settings["max_joint_deviation_deg"]=20.;
    CalibrationGuard relaxed(relaxed_settings);relaxed.arm("relaxed",pose,joints);
    auto relaxed_q=joints;relaxed_q[0]=CalibrationGuard::radians(20.);
    relaxed.check_arm(relaxed_q);
    expect_failure([&]{relaxed_q[0]=CalibrationGuard::radians(20.1);relaxed.check_arm(relaxed_q);});
    expect_failure([&]{auto invalid=data;invalid["max_joint_deviation_deg"]=20.1;CalibrationGuard refused(invalid);});
    auto tilt=pose;
    Eigen::Quaterniond rotation(Eigen::AngleAxisd(CalibrationGuard::radians(4),Eigen::Vector3d::UnitX()));
    tilt[3]=rotation.x();tilt[4]=rotation.y();tilt[5]=rotation.z();tilt[6]=rotation.w();
    expect_failure([&]{guard.target("test",tilt,now);});
    guard.target("test",pose,now);
    guard.tick(pose,joints,zero,zero,now);
    expect_failure([&]{guard.tick(pose,joints,zero,zero,now+std::chrono::seconds(1));});
    auto fast=zero;fast[0]=1;
    expect_failure([&]{guard.tick(pose,joints,fast,zero,now);});
    expect_failure([&]{guard.tick(pose,joints,zero,fast,now);});
    auto changed=joints;changed[0]=1;
    expect_failure([&]{guard.tick(pose,changed,zero,zero,now);});
    auto minor=zero;minor[0]=guard.value("max_linear_speed_m_s")*1.3;
    guard.tick(pose,joints,minor,zero,now);
    if(guard.status().at("speed_warning").get<std::string>().empty()||!guard.fault.empty())throw std::runtime_error("Minor TCP overspeed must warn without latching");
    expect_failure([&]{auto severe=minor;severe[0]=guard.value("max_linear_speed_m_s")*2.1;guard.tick(pose,joints,severe,zero,now);});
    auto joint_warning=zero;joint_warning[0]=CalibrationGuard::radians(guard.value("max_joint_speed_deg_s")*1.3);
    guard.tick(pose,joints,zero,joint_warning,now);
    if(guard.speed_warning.empty())throw std::runtime_error("Minor joint overspeed must warn");
    expect_failure([&]{auto severe=joint_warning;severe[0]=CalibrationGuard::radians(guard.value("max_joint_speed_deg_s")*1.6);guard.tick(pose,joints,zero,severe,now);});
    guard.tick(pose,joints,zero,zero,now);
    if(!guard.speed_warning.empty())throw std::runtime_error("Speed warning did not clear");
    expect_failure([&]{auto invalid=data;invalid["measured_tcp_speed_trip_factor"]=2.1;CalibrationGuard refused(invalid);});
    Eigen::Matrix<double,6,1> previous=Eigen::Matrix<double,6,1>::Zero();
    for(int i=0;i<1000;++i) {
        auto next=guard.limit_speed(Eigen::Matrix<double,6,1>::Ones()*100,previous,500);
        if(next.head<3>().norm()>.0100001 || next.tail<3>().norm()>CalibrationGuard::radians(2.00001)) throw std::runtime_error("Speed cap failure");
        if((next-previous).head<3>().norm()>.02/500+1e-10 || (next-previous).tail<3>().norm()>CalibrationGuard::radians(4)/500+1e-10) throw std::runtime_error("Acceleration cap failure");
        previous=next;
    }
    if(argc==3){
        auto plane_settings=data;
        for(const std::string key:{"workspace_min_m","workspace_max_m","table_min_xy_m","table_max_xy_m","table_top_z_m","tool_radius_m","arm_link_radii_m"})plane_settings.erase(key);
        plane_settings["geometry_mode"]="plane_only";plane_settings["allow_estimated_plane"]=true;
        plane_settings["verified"]=false;plane_settings["robot_safety_setup_verified"]=false;
        plane_settings["collision_model_path"]=argv[2];
        CalibrationGuard plane(plane_settings);
        expect_failure([&]{plane.arm("plane",pose,joints);});
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::system_clock::now().time_since_epoch()).count();
        nlohmann::json feedback={{"hand_q",zero},{"hand_timestamp_ms",ms}};
        expect_failure([&]{plane.update_hand(feedback);}); // six angles cannot masquerade as a hand
        feedback["hand_q"]=std::vector<double>(20,0.);
        plane.update_hand(feedback);plane.arm("plane",pose,joints);
        plane.target("plane",pose,CalibrationGuard::Clock::now());
        plane.tick(pose,joints,zero,zero,CalibrationGuard::Clock::now());
        expect_failure([&]{plane.tick(pose,joints,zero,zero,CalibrationGuard::Clock::now()+std::chrono::seconds(1));});
        expect_failure([&]{auto stale=feedback;stale["hand_timestamp_ms"]=ms-1000;plane.update_hand(stale);});
        expect_failure([&]{auto bad=plane_settings;bad["allow_estimated_plane"]=false;CalibrationGuard refused(bad);});
        expect_failure([&]{auto bad=plane_settings;bad["table_plane"]["offset_m"]=-.2;CalibrationGuard collision(bad);collision.update_hand(feedback);collision.arm("collision",pose,joints);});
        std::cout<<"Plane-only: estimated configuration, independent URDF surfaces and hand-feedback expiry checks passed.\n";
    }
    guard.trip("operator stop");
    expect_failure([&]{guard.arm("test",pose,joints);});
    expect_failure([&]{guard.target("test",pose,now);});
    std::cout<<"Calibration guard: geometry, arm/table collision, session, timeout, speed, acceleration and fault-latch checks passed.\n";
}
